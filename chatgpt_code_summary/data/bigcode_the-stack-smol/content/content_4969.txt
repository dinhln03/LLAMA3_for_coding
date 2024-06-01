# -*- coding: utf-8 -*-
"""
solutions_by_text.sbt_token_generator
~~~~~~~~~~~~
This module handles security token generation.
"""
# @Author: sijanonly
# @Date:   2018-03-19 10:57:26
# @Last Modified by:   sijanonly
# @Last Modified time: 2018-03-19 14:51:07

import json

from urllib import parse

import requests


from .handle_exceptions import CustomException

_base_url = 'https://{}.solutionsbytext.com/SBT.App.SetUp/RSServices/'


def create_security_token(api_key, stage):
    """
    Generates a security token for SBT API access.

    Args:
        api_key (string): API_KEY value provided by solutionsbytext
        stage (string): STAGE values (test or ui)

    Returns:
        string: SecurityToken returns by LoginAPIService

    Raises:
        CustomException: Raises while error during GET request.

    """

    url = ''.join(
        [
            _base_url.format(stage),
            'LoginAPIService.svc/AuthenticateAPIKey?',
            parse.urlencode({'APIKey': api_key})
        ]
    )

    response_data = json.loads(requests.get(url).text)
    if response_data['AuthenticateAPIKeyResult'].get('ErrorCode') == 1402:
        raise CustomException(
            'Error in generating security key.')

    if response_data['AuthenticateAPIKeyResult'].get('ErrorCode') == 1401:
        raise CustomException(
            'SecurityToken generation is failed.')

    return response_data['AuthenticateAPIKeyResult'].get('SecurityToken')
