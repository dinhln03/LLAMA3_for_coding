"""
This module contains the lambda function code for put-storage-tags API.

This file uses environment variables in place of config; thus
sddcapi_boot_dir is not required.
"""

# pylint: disable=import-error,logging-format-interpolation,broad-except,too-many-statements,C0413,W1203,R1703,R0914
import boto3
import botocore.exceptions
import os
import sys
import json
import traceback
from cloudx_sls_authorization import lambda_auth

THISDIR = os.path.dirname(__file__)  # boto3-proxy
APPDIR = os.path.dirname(THISDIR)  # boto3-proxy

if APPDIR not in sys.path:
    sys.path.append(APPDIR)
if THISDIR not in sys.path:
    sys.path.append(THISDIR)

from utils import api_request
from utils import helpers, secrets
from utils.exceptions import InvalidRegionException, InvalidInputException
from utils.log_helper import Logger

logger = Logger()

# Define LDAP lookup configs
LDAP_SERVER = os.environ['LDAP_SERVER']
LDAP_USERNAME = os.environ['LDAP_USERNAME']
LDAP_PASSWORD_SECRET_NAME = os.environ['LDAP_PASSWORD_SECRET_NAME']
LDAP_SEARCH_BASE = os.environ['LDAP_SEARCH_BASE']
LDAP_OBJECT_CLASS = os.environ['LDAP_OBJECT_CLASS']
LDAP_GROUP_NAME = os.environ['LDAP_GROUP_NAME'].split(',')
LDAP_LOOKUP_ATTRIBUTE = os.environ['LDAP_LOOKUP_ATTRIBUTE']
MSFT_IDP_TENANT_ID = os.environ['MSFT_IDP_TENANT_ID']
MSFT_IDP_APP_ID = os.environ['MSFT_IDP_APP_ID'].split(',')
MSFT_IDP_CLIENT_ROLES = os.environ['MSFT_IDP_CLIENT_ROLES'].split(',')

# Status codes
SUCCESS_STATUS = 200  # success
BAD_REQUEST_STATUS = 400  # service not supported, action not supported
NOT_FOUND_STATUS = 404  # invalid account, invalid region
UNAUTHORIZED_STATUS = 401  # invalid auth token
INTERNAL_SERVICE_ERROR_STATUS = 500  # internal service error


def handler(event, context):
    """
    Boto3 Proxy API Handler
    """
    headers = event.get('Headers', event.get('headers'))
    if 'request-context-id' in headers:
        logger.set_uuid(headers['request-context-id'])
    logger.info({"Incoming event": event})
    logger.info('Incoming context: %s', context)

    request_body = json.loads(event.get('body', {}))

    try:
        # Define service client
        secrets_client = boto3.client('secretsmanager')
        lambda_auth.authorize_lambda_request(event, MSFT_IDP_TENANT_ID, MSFT_IDP_APP_ID,
                                             MSFT_IDP_CLIENT_ROLES, LDAP_SERVER, LDAP_USERNAME,
                                             secrets.retrieve_ldap_password(secrets_client,
                                                                            logger,
                                                                            LDAP_PASSWORD_SECRET_NAME
                                                                            ),
                                             LDAP_SEARCH_BASE,
                                             LDAP_OBJECT_CLASS, LDAP_GROUP_NAME, LDAP_LOOKUP_ATTRIBUTE)

        # Get the SSM client
        ssm_client = boto3.client('ssm')

    except Exception as e:
        traceback.print_exc()
        return {
            'statusCode': UNAUTHORIZED_STATUS,
            'body': json.dumps({'error': f"Unauthorized. {str(e)}"})
        }

    # Get environment variables
    resp_headers = {
        'Content-Type': 'application/json',
        "request-context-id": logger.get_uuid()
    }
    if hasattr(context, 'local_test'):
        logger.info('Running at local')
    path_params = event.get('pathParameters', {})
    request_headers = event.get('headers', {})
    vpcxiam_endpoint = os.environ.get('vpcxiam_endpoint')
    vpcxiam_scope = os.environ.get('vpcxiam_scope')
    vpcxiam_host = os.environ.get('vpcxiam_host')

    # Set the default success response and status code
    resp = {
        'message': 'API action has been successfully completed'
    }
    status_code = SUCCESS_STATUS

    try:
        account = path_params.get('account-id')
        region = path_params.get('region-name')
        service = path_params.get('boto3-service')
        action = path_params.get('boto3-action')
        logger.info(f"Account: {account}")
        logger.info(f"Region: {region}")
        logger.info(f"Boto3 Service: {service}")
        logger.info(f"Boto3 Action: {action}")

        # is authorized?
        logger.info(f'is_authorized({request_headers}, {MSFT_IDP_APP_ID}, '
                    f'{MSFT_IDP_TENANT_ID}, {MSFT_IDP_CLIENT_ROLES}')

        # Get the credentials for the account resources will be created in.
        url = (vpcxiam_endpoint +
               f"/v1/accounts/{account}/roles/admin/credentials")
        scope = vpcxiam_scope
        additional_headers = {
            'Host': vpcxiam_host
        }
        api_requests = api_request.ApiRequests()
        credentials = json.loads(
            (api_requests.request(url=url, method='get', scope=scope, additional_headers=additional_headers)).text
        )
        error = credentials.get('error', {})
        if error:
            logger.error(error)
            raise ValueError(error)
        credentials = credentials.get('credentials', {})
        try:
            # Validate service and if valid, get the allowed actions
            ssm_parameter_name = '/vpcx/aws/boto3-proxy/allowed-actions/'+service
            logger.info("Looking up parameter "+ssm_parameter_name)
            allowed_actions = ssm_client.get_parameter(Name=ssm_parameter_name)
        except botocore.exceptions.ClientError as err:
            logger.error(err)
            if err.response['Error']['Code'] == 'ParameterNotFound':
                raise InvalidInputException("Service " + service + " is not an allowed service for the API")
            else:
                raise error

        # Validate action
        if action not in allowed_actions['Parameter']['Value']:
            raise InvalidInputException("Action "+action+" is not an allowed action for the API")

        # Validate region
        ec2_client = boto3.client(
            service_name='ec2',
            aws_access_key_id=credentials.get('AccessKeyId', ''),
            aws_secret_access_key=credentials.get('SecretAccessKey', ''),
            aws_session_token=credentials.get('SessionToken', ''))
        helpers.is_region_valid(ec2_client, region)
        logger.info(f"{region} is a valid region")

        # Create clients for the given region and given account
        boto3_client = boto3.client(
            service_name=service,
            region_name=region,
            aws_access_key_id=credentials.get('AccessKeyId', ''),
            aws_secret_access_key=credentials.get('SecretAccessKey', ''),
            aws_session_token=credentials.get('SessionToken', ''))

        # Call the action (function) for the boto3 client's service with the request params
        kwargs = request_body
        getattr(boto3_client, action)(**kwargs)

    # boto3 error
    except botocore.exceptions.ClientError as err:
        status_code = INTERNAL_SERVICE_ERROR_STATUS
        resp = {
            'error': f'{type(err).__name__}: {err}'
        }
    except InvalidRegionException:
        status_code = NOT_FOUND_STATUS
        resp = {
            'error': 'Please enter a valid region in the url path'
        }
    except InvalidInputException as err:
        status_code = BAD_REQUEST_STATUS
        resp = {
            'error': str(err)
        }
    except ValueError as err:
        status_code = NOT_FOUND_STATUS
        resp = {
            'error': str(err)
        }
    except Exception as err:
        status_code = INTERNAL_SERVICE_ERROR_STATUS
        resp = {
            'error': f'{type(err).__name__}: {err}'
        }
    resp = helpers.lambda_returns(status_code, resp_headers, json.dumps(resp))
    logger.info(f'response: {resp}')
    return resp
