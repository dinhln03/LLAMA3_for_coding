__author__ = 'tinglev'

import logging
import requests
from requests import HTTPError, ConnectTimeout, RequestException
from modules import environment
from modules.subscribers.slack import slack_util
from modules.event_system.event_system import subscribe_to_event, unsubscribe_from_event
from modules import deployment_util

LOG = logging.getLogger(__name__)

DEFAULT_FLOTTSBRO_API_BASE_URL = 'https://api-r.referens.sys.kth.se/api/pipeline'

def subscribe():
    subscribe_to_event('deployment', handle_deployment)

def unsubscribe():
    unsubscribe_from_event('deployment', handle_deployment)

def handle_deployment(deployment):
    global LOG
    add(deployment)
    return deployment

def get_base_url():
    return environment.get_env_with_default_value(environment.FLOTTSBRO_API_BASE_URL, DEFAULT_FLOTTSBRO_API_BASE_URL)

def get_add_endpoint(cluster):
    return '{}/v1/latest/{}'.format(get_base_url(), cluster)

def add(deployment):
    call_endpoint(get_add_endpoint(deployment["cluster"]), deployment)

def get_headers():
    api_key = environment.get_env(environment.FLOTTSBRO_API_KEY)
    if not api_key:
        LOG.error('No header env FLOTTSBRO_API_KEY specified ')
        return None

    return {
            'api_key':  api_key
        }

def call_endpoint(endpoint, deployment):
    global LOG

    try:
        headers = get_headers()
        if headers:
            response = requests.post(endpoint, data=deployment, headers=headers)
            LOG.debug('Calling "%s", response was "%s"', endpoint, response.text)
        else:
            LOG.info('Skipped calling flottsbro-api, header constraints not satisfied.')

    except (HTTPError, ConnectTimeout, RequestException) as request_ex:
        LOG.error('Could not add deployment to Flottsbro-API: "%s"', request_ex)