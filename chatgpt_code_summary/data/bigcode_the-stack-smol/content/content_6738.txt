'''
Preprocessor for Foliant documentation authoring tool.

Calls Elasticsearch API to generate an index based on Markdown content.
'''

import re
import json
from os import getenv
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from markdown import markdown
from bs4 import BeautifulSoup

from foliant.preprocessors.base import BasePreprocessor


class Preprocessor(BasePreprocessor):
    defaults = {
        'es_url': 'http://127.0.0.1:9200/',
        'index_name': '',
        'index_copy_name': '',
        'index_properties': {},
        'actions': [
            'delete',
            'create'
        ],
        'use_chapters': True,
        'format': 'plaintext',
        'escape_html': True,
        'url_transform': [
            {'\/?index\.md$': '/'},
            {'\.md$': '/'},
            {'^([^\/]+)': '/\g<1>'}
        ],
        'require_env': False,
        'targets': []
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = self.logger.getChild('elasticsearch')

        self.logger.debug(f'Preprocessor inited: {self.__dict__}')

    def _get_url(self, markdown_file_path: str) -> str:
        url = str(markdown_file_path.relative_to(self.working_dir))
        url_transformation_rules = self.options['url_transform']

        if not isinstance(url_transformation_rules, list):
            url_transformation_rules = [url_transformation_rules]

        for url_transformation_rule in url_transformation_rules:
            for pattern, replacement in url_transformation_rule.items():
                url = re.sub(pattern, replacement, url)

        return url

    def _get_title(self, markdown_content: str) -> str or None:
        headings_found = re.search(
            r'^\#{1,6}\s+(.+?)(?:\s+\{\#\S+\})?\s*$',
            markdown_content,
            flags=re.MULTILINE
        )

        if headings_found:
            return headings_found.group(1)

        return None

    def _get_chapters_paths(self) -> list:
        def _recursive_process_chapters(chapters_subset):
            if isinstance(chapters_subset, dict):
                processed_chapters_subset = {}

                for key, value in chapters_subset.items():
                    processed_chapters_subset[key] = _recursive_process_chapters(value)

            elif isinstance(chapters_subset, list):
                processed_chapters_subset = []

                for item in chapters_subset:
                    processed_chapters_subset.append(_recursive_process_chapters(item))

            elif isinstance(chapters_subset, str):
                if chapters_subset.endswith('.md'):
                    chapters_paths.append(self.working_dir / chapters_subset)

                processed_chapters_subset = chapters_subset

            else:
                processed_chapters_subset = chapters_subset

            return processed_chapters_subset

        chapters_paths = []
        _recursive_process_chapters(self.config['chapters'])

        self.logger.debug(f'Chapters files paths: {chapters_paths}')

        return chapters_paths

    def _http_request(
        self,
        request_url: str,
        request_method: str = 'GET',
        request_headers: dict or None = None,
        request_data: bytes or None = None
    ) -> dict:
        http_request = request.Request(request_url, method=request_method)

        if request_headers:
            http_request.headers = request_headers

        if request_data:
            http_request.data = request_data

        try:
            with request.urlopen(http_request) as http_response:
                response_status = http_response.getcode()
                response_headers = http_response.info()
                response_data = http_response.read()

        except HTTPError as http_response_not_ok:
            response_status = http_response_not_ok.getcode()
            response_headers = http_response_not_ok.info()
            response_data = http_response_not_ok.read()

        return {
            'status': response_status,
            'headers': response_headers,
            'data': response_data
        }

    def _escape_html(self, content: str) -> str:
        return content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

    def _create_index(self, index_name: str) -> None:
        if self.options['index_properties']:
            create_request_url = f'{self.options["es_url"].rstrip("/")}/{index_name}/'

            self.logger.debug(
                'Calling Elasticsearch API to create an index with specified properties, ' +
                f'URL: {create_request_url}'
            )

            create_response = self._http_request(
                create_request_url,
                'PUT',
                {
                    'Content-Type': 'application/json; charset=utf-8'
                },
                json.dumps(self.options['index_properties'], ensure_ascii=False).encode('utf-8')
            )

            create_response_data = json.loads(create_response['data'].decode('utf-8'))

            self.logger.debug(f'Response received, status: {create_response["status"]}')
            self.logger.debug(f'Response headers: {create_response["headers"]}')
            self.logger.debug(f'Response data: {create_response_data}')

            if create_response['status'] == 200 and create_response_data.get('acknowledged', None) is True:
                self.logger.debug('Index created')

            elif create_response['status'] == 400 and create_response_data.get(
                'error', {}
            ).get(
                'type', ''
            ) == 'resource_already_exists_exception':
                self.logger.debug('Index already exists')

            else:
                error_message = 'Failed to create an index'
                self.logger.error(f'{error_message}')
                raise RuntimeError(f'{error_message}')

        else:
            self.logger.debug('An index without specific properties will be created')

        if self.options['use_chapters']:
            self.logger.debug('Only files mentioned in chapters will be indexed')

            markdown_files_paths = self._get_chapters_paths()

        else:
            self.logger.debug('All files of the project will be indexed')

            markdown_files_paths = self.working_dir.rglob('*.md')

        data_for_indexing = ''

        for markdown_file_path in markdown_files_paths:
            self.logger.debug(f'Processing the file: {markdown_file_path}')

            with open(markdown_file_path, encoding='utf8') as markdown_file:
                markdown_content = markdown_file.read()

            if markdown_content:
                url = self._get_url(markdown_file_path)
                title = self._get_title(markdown_content)

                if self.options['format'] == 'html' or self.options['format'] == 'plaintext':
                    self.logger.debug(f'Converting source Markdown content to: {self.options["format"]}')

                    content = markdown(markdown_content)

                    if self.options['format'] == 'plaintext':
                        soup = BeautifulSoup(content, 'lxml')

                        for non_text_node in soup(['style', 'script']):
                            non_text_node.extract()

                        content = soup.get_text()

                        if self.options['escape_html']:
                            self.logger.debug('Escaping HTML syntax')

                            if title:
                                title = self._escape_html(title)

                            content = self._escape_html(content)

                else:
                    self.logger.debug('Leaving source Markdown content unchanged')

                    content = markdown_content

                self.logger.debug(f'Adding the page, URL: {url}, title: {title}')

                data_for_indexing += '{"index": {}}\n' + json.dumps(
                    {
                        'url': url,
                        'title': title,
                        'content': content
                    },
                    ensure_ascii=False
                ) + '\n'

            else:
                self.logger.debug('It seems that the file has no content')

        self.logger.debug(f'Data for indexing: {data_for_indexing}')

        update_request_url = f'{self.options["es_url"].rstrip("/")}/{index_name}/_bulk?refresh'

        self.logger.debug(f'Calling Elasticsearch API to add the content to the index, URL: {update_request_url}')

        update_response = self._http_request(
            update_request_url,
            'POST',
            {
                'Content-Type': 'application/json; charset=utf-8'
            },
            data_for_indexing.encode('utf-8')
        )

        update_response_data = json.loads(update_response['data'].decode('utf-8'))

        self.logger.debug(f'Response received, status: {update_response["status"]}')
        self.logger.debug(f'Response headers: {update_response["headers"]}')
        self.logger.debug(f'Response data: {update_response_data}')

        if update_response['status'] != 200 or update_response_data.get('errors', True):
            error_message = 'Failed to add content to the index'
            self.logger.error(f'{error_message}')
            raise RuntimeError(f'{error_message}')

        return None

    def _delete_index(self, index_name: str) -> None:
        delete_request_url = f'{self.options["es_url"].rstrip("/")}/{index_name}/'

        self.logger.debug(f'Calling Elasticsearch API to delete the index, URL: {delete_request_url}')

        delete_response = self._http_request(
            delete_request_url,
            'DELETE'
        )

        delete_response_data = json.loads(delete_response['data'].decode('utf-8'))

        self.logger.debug(f'Response received, status: {delete_response["status"]}')
        self.logger.debug(f'Response headers: {delete_response["headers"]}')
        self.logger.debug(f'Response data: {delete_response_data}')

        if delete_response['status'] == 200 and delete_response_data.get('acknowledged', None) is True:
            self.logger.debug('Index deleted')

        elif delete_response['status'] == 404 and delete_response_data.get(
            'error', {}
        ).get(
            'type', ''
        ) == 'index_not_found_exception':
            self.logger.debug('Index does not exist')

        else:
            error_message = 'Failed to delete the index'
            self.logger.error(f'{error_message}')
            raise RuntimeError(f'{error_message}')

        return None

    def _update_index_setting(self, index_name: str, settings_to_update: dict) -> None:
        update_request_url = f'{self.options["es_url"].rstrip("/")}/{index_name}/_settings/'

        self.logger.debug(f'Calling Elasticsearch API to update the index settings, URL: {update_request_url}')

        update_response = self._http_request(
            update_request_url,
            'PUT',
            {
                'Content-Type': 'application/json; charset=utf-8'
            },
            json.dumps(
                settings_to_update,
                ensure_ascii=False
            ).encode('utf-8')
        )

        update_response_data = json.loads(update_response['data'].decode('utf-8'))

        self.logger.debug(f'Response received, status: {update_response["status"]}')
        self.logger.debug(f'Response headers: {update_response["headers"]}')
        self.logger.debug(f'Response data: {update_response_data}')

        if update_response['status'] == 200 and update_response_data.get('acknowledged', None) is True:
            self.logger.debug('Index settings updated')

        else:
            error_message = 'Failed to update the index settings'
            self.logger.error(f'{error_message}')
            raise RuntimeError(f'{error_message}')

        return None

    def _clone_index(self, index_name: str, index_copy_name: str) -> None:
        clone_request_url = f'{self.options["es_url"].rstrip("/")}/{index_name}/_clone/{index_copy_name}/'

        self.logger.debug(f'Calling Elasticsearch API to clone the index, URL: {clone_request_url}')

        clone_response = self._http_request(
            clone_request_url,
            'POST'
        )

        clone_response_data = json.loads(clone_response['data'].decode('utf-8'))

        self.logger.debug(f'Response received, status: {clone_response["status"]}')
        self.logger.debug(f'Response headers: {clone_response["headers"]}')
        self.logger.debug(f'Response data: {clone_response_data}')

        if clone_response['status'] == 200 and clone_response_data.get('acknowledged', None) is True:
            self.logger.debug('Index cloned')

        else:
            error_message = 'Failed to clone the index'
            self.logger.error(f'{error_message}')
            raise RuntimeError(f'{error_message}')

        return None

    def _copy_index(self, index_name: str, index_copy_name: str) -> None:
        if not index_copy_name:
            index_copy_name = index_name + '_copy'

        self.logger.debug(f'Copying the index {index_name} to {index_copy_name}')

        self.logger.debug(f'First, marking the index {index_name} as read-only')

        self._update_index_setting(
            index_name,
            {
                'settings': {
                    'index.blocks.write': True
                }
            }
        )

        self.logger.debug(f'Second, deleting the index {index_copy_name}, if exists')

        self._delete_index(index_copy_name)

        self.logger.debug(f'Third, cloning the index {index_name} as {index_copy_name}')

        self._clone_index(index_name, index_copy_name)

        self.logger.debug(f'Fourth, unmarking the index {index_name} as read-only')

        self._update_index_setting(
            index_name,
            {
                'settings': {
                    'index.blocks.write': False
                }
            }
        )

        self.logger.debug(f'Fifth, also unmarking the index {index_copy_name} as read-only')

        self._update_index_setting(
            index_copy_name,
            {
                'settings': {
                    'index.blocks.write': False
                }
            }
        )

        return None

    def apply(self):
        self.logger.info('Applying preprocessor')

        envvar = 'FOLIANT_ELASTICSEARCH'

        if not self.options['require_env'] or getenv(envvar) is not None:
            self.logger.debug(
                f'Allowed targets: {self.options["targets"]}, ' +
                f'current target: {self.context["target"]}'
            )

            if not self.options['targets'] or self.context['target'] in self.options['targets']:
                actions = self.options['actions']

                if not isinstance(self.options['actions'], list):
                    actions = [actions]

                for action in actions:
                    self.logger.debug(f'Applying action: {action}')

                    if action == 'create':
                        self._create_index(self.options['index_name'])

                    elif action == 'delete':
                        self._delete_index(self.options['index_name'])

                    elif action == 'copy':
                        self._copy_index(self.options['index_name'], self.options['index_copy_name'])

                    else:
                        self.logger.debug('Unknown action, skipping')

        else:
            self.logger.debug(f'Environment variable {envvar} is not set, skipping')

        self.logger.info('Preprocessor applied')
