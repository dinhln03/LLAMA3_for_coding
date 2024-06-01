import datetime
from typing import Dict, Tuple, Any

import boto3
from botocore.stub import Stubber
from dateutil.tz import tzutc

from dassana.common.aws_client import LambdaTestContext
from json import dumps
import pytest


@pytest.fixture()
def input_s3_with_website(s3_public_bucket_with_website, region):
    return {
        'bucketName': s3_public_bucket_with_website,
        'region': region
    }


@pytest.fixture()
def iam_policy():
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "ec2:GetDefaultCreditSpecification",
                    "ec2:GetEbsEncryptionByDefault",
                    "ec2:ExportClientVpnClientConfiguration",
                    "ec2:GetCapacityReservationUsage",
                    "ec2:DescribeVolumesModifications",
                    "ec2:GetHostReservationPurchasePreview",
                    "ec2:GetSubnetCidrReservations",
                    "ec2:GetConsoleScreenshot",
                    "ec2:GetConsoleOutput",
                    "ec2:ExportClientVpnClientCertificateRevocationList",
                    "ec2:GetLaunchTemplateData",
                    "ec2:GetSerialConsoleAccessStatus",
                    "ec2:GetFlowLogsIntegrationTemplate",
                    "ec2:DescribeScheduledInstanceAvailability",
                    "ec2:GetEbsDefaultKmsKeyId",
                    "ec2:GetManagedPrefixListEntries",
                    "ec2:DescribeVpnConnections",
                    "ec2:DescribeTags",
                    "ec2:GetCoipPoolUsage",
                    "ec2:DescribeFastSnapshotRestores",
                    "ec2:GetReservedInstancesExchangeQuote",
                    "ec2:GetAssociatedEnclaveCertificateIamRoles",
                    "ec2:GetPasswordData",
                    "ec2:GetAssociatedIpv6PoolCidrs",
                    "ec2:DescribeScheduledInstances",
                    "ec2:GetManagedPrefixListAssociations",
                    "ec2:DescribeElasticGpus"
                ],
                "Resource": "*"
            }
        ]
    }


@pytest.fixture()
def iam_role_name():
    return 'ec2-iam-role'


@pytest.fixture()
def instance_profile_name():
    return 'ec2-instance-profile-role'


@pytest.fixture()
def iam_role_arn(iam_client, iam_policy, iam_role_name, instance_profile_name) -> Tuple[Any, Dict[str, Any]]:
    resp = iam_client.create_role(RoleName=iam_role_name, AssumeRolePolicyDocument=dumps(iam_policy))
    instance_profile_resp = iam_client.create_instance_profile(
        InstanceProfileName=instance_profile_name
    )
    iam_client.add_role_to_instance_profile(
        InstanceProfileName=instance_profile_name,
        RoleName=iam_role_name
    )

    instance_profile_resp = instance_profile_resp.get('InstanceProfile')
    return resp['Role']['Arn'], {
        'Name': instance_profile_resp.get('InstanceProfileName'),
        'Arn': instance_profile_resp.get('Arn')
    }


@pytest.fixture()
def ec2_instance_with_role(ec2_client, iam_role_arn, instance_profile_name):
    instances = ec2_client.run_instances(ImageId='ami-1234',
                                         MinCount=1,
                                         MaxCount=1,
                                         InstanceType='t2.micro',
                                         IamInstanceProfile=iam_role_arn[1])
    instance_id = instances.get('Instances')[0].get('InstanceId')
    assoc_resp = ec2_client.associate_iam_instance_profile(IamInstanceProfile=iam_role_arn[1], InstanceId=instance_id)
    return instance_id


@pytest.fixture()
def ec2_instance_without_role(ec2_client):
    ec2_client.run_instances(ImageId='ami-1234-foobar',
                             MinCount=1,
                             MaxCount=1)
    instances = ec2_client.describe_instances(
        Filters=[
            {
                'Name': 'image-id',
                'Values': ['ami-1234-foobar']
            }
        ]
    )['Reservations'][0]['Instances']
    return instances[0]['InstanceId']


def test_ec2_instance_with_role(ec2_instance_with_role, iam_role_arn, region):
    from handler_ec2_role import handle

    result: Dict = handle({'instanceId': ec2_instance_with_role, 'region': region},
                          LambdaTestContext('foobar', env={},
                                            custom={}))
    assert result.get('result').get('roleName') == iam_role_arn[1].get('Name')
    assert str.split(result.get('result').get('roleArn'), ':role/') == str.split(iam_role_arn[1].get(
        'Arn'), ':instance-profile/')


def test_ec2_instance_without_role(ec2_instance_without_role, region):
    from handler_ec2_role import handle
    result: Dict = handle({'instanceId': ec2_instance_without_role, 'region': region},
                          LambdaTestContext('foobar', env={}, custom={}))
    assert result.get('result').get('roleArn') == ''
    assert result.get('result').get('roleName') == ''


def test_ec2_instance_does_not_exist(ec2_instance_without_role, region):
    from handler_ec2_role import handle
    result: Dict = handle({'instanceId': 'i-abcd', 'region': region},
                          LambdaTestContext('foobar', env={}, custom={}))
    assert result.get('result').get('roleArn') == ''
    assert result.get('result').get('roleName') == ''
