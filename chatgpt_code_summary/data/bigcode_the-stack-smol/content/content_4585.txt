#
# Copyright 2021 XEBIALABS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import json
import requests
import org.slf4j.LoggerFactory as LoggerFactory

logger = LoggerFactory.getLogger("Arxan")

# New ARXAN logic
# setup the request url
api_token_endpoint = "/v2/apaas/apps"
url = server.get('url') + "%s" % api_token_endpoint

headers = {
    'Content-Type': "application/x-www-form-urlencoded"
}

with open(file_path, 'rb') as app_file:
    logger.info('Filepath: %s' % file_path)
    files = {'appFile': app_file}
    headers = {
        'Authorization': auth_string,
    }
    data = {
        'productId' : 'Essential Protection',
        'protection': {
            'appAware': {
                'applicationToken': server.get('app_token'),
                'endpoint': server.get('app_endpoint')
            }
        }
    }
    logger.info('Uploading file...')
    logger.info('URL: %s' % url)
    logger.info('Headers: %s' % json.dumps(headers))
    logger.info('JSON: %s' % json.dumps(data))
    response = requests.post(url, files = files, data = {'data': json.dumps(data)}, headers = headers, verify = False)
    logger.info('Uploading app response status code: %s.' % response.status_code)
    logger.info(response.json()['message'])
    # output = response.json().get('protectionId')

    if response.status_code == 200:
        logger.info('App uploaded')
        json_response = response.json()
        logger.debug('App upload response: %s', json_response)
        if 'protectionId' not in json_response:
            logger.error('There was a problem uploading the app. Missing protectionId in the response')
        else:
            protection_id = json_response['protectionId']
            logger.debug('App protection id is %s', protection_id)
            output = protection_id
    elif response.status_code == 400:
        error_message = response.json()['message']
        logger.error('There was a problem protecting %s', error_message)
    elif response.status_code == 401 or response.status_code == 403:
        raise AuthorizationError()
    elif response.status_code == 404:
        logger.error('Cannot reach server %s', server)
    else:
        logger.error('An unexpected error has occurred. (%d)', response.status_code)
        raise Exception('Incorrect response code for upload app: (%s)', response.status_code)