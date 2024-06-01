# -*- coding: utf-8 -*-
import time
import mock
import pytest
import requests

try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin

from anacode import codes
from anacode.api import client
from anacode.api import writers


def empty_response(*args, **kwargs):
    resp = requests.Response()
    resp._content = b'{}'
    resp.status_code = 200
    return resp


def empty_json(*args, **kwargs):
    return {}


@pytest.fixture
def auth():
    return '1234567890'


@pytest.fixture
def auth_header(auth):
    return {'Authorization': 'Token %s' % auth,
            'Accept': 'application/json'}


@pytest.fixture
def api(auth):
    return client.AnacodeClient(auth)


@mock.patch('requests.post', empty_response)
def test_scrape_call(api, auth_header, mocker):
    mocker.spy(requests, 'post')
    api.scrape('http://chinese.portal.com.ch')
    assert requests.post.call_count == 1
    requests.post.assert_called_once_with(
        urljoin(client.ANACODE_API_URL, 'scrape/'),
        headers=auth_header, json={'url': 'http://chinese.portal.com.ch'})


@mock.patch('requests.post', empty_response)
def test_categories_call(api, auth_header, mocker):
    mocker.spy(requests, 'post')
    api.analyze(['安全性能很好，很帅气。'], ['categories'])
    assert requests.post.call_count == 1
    json_data = {'texts': ['安全性能很好，很帅气。'], 'analyses': ['categories']}
    requests.post.assert_called_once_with(
        urljoin(client.ANACODE_API_URL, 'analyze/'),
        headers=auth_header, json=json_data)


@mock.patch('requests.post', empty_response)
def test_sentiment_call(api, auth_header, mocker):
    mocker.spy(requests, 'post')
    api.analyze(['安全性能很好，很帅气。'], ['sentiment'])
    assert requests.post.call_count == 1
    requests.post.assert_called_once_with(
        urljoin(client.ANACODE_API_URL, 'analyze/'),
        headers=auth_header, json={'texts': ['安全性能很好，很帅气。'],
                                   'analyses': ['sentiment']})


@mock.patch('requests.post', empty_response)
def test_concepts_call(api, auth_header, mocker):
    mocker.spy(requests, 'post')
    api.analyze(['安全性能很好，很帅气。'], ['concepts'])
    assert requests.post.call_count == 1
    requests.post.assert_called_once_with(
        urljoin(client.ANACODE_API_URL, 'analyze/'),
        headers=auth_header, json={'texts': ['安全性能很好，很帅气。'],
                                   'analyses': ['concepts']})


@mock.patch('requests.post', empty_response)
def test_absa_call(api, auth_header, mocker):
    mocker.spy(requests, 'post')
    api.analyze(['安全性能很好，很帅气。'], ['absa'])
    assert requests.post.call_count == 1
    requests.post.assert_called_once_with(
        urljoin(client.ANACODE_API_URL, 'analyze/'),
        headers=auth_header, json={'texts': ['安全性能很好，很帅气。'],
                                   'analyses': ['absa']})


@pytest.mark.parametrize('code,call,args', [
    (codes.SCRAPE, 'scrape', ['http://www.google.com/']),
    (codes.ANALYZE, 'analyze', ['安全性能很好，很帅气。', ['categories']]),
])
def test_proper_method_call(api, code, call, args, mocker):
    mock.patch('anacode.api.client.AnacodeClient.' + call, empty_json)
    mocker.spy(api, call)
    api.call((code, *args))
    getattr(api, call).assert_called_once_with(*args)


@pytest.mark.parametrize('call,args', [
    ('scrape', ['http://www.google.com/']),
    ('analyze', [['安全性能很好，很帅气。'], ['categories', 'concepts']]),
])
@pytest.mark.parametrize('count,call_count', [
    (0, 0), (5, 0), (9, 0), (10, 1), (11, 1), (19, 1), (20, 2),
])
def test_should_start_analysis(api, mocker, call, args, count, call_count):
    writer = writers.DataFrameWriter()
    writer.init()

    to_mock = 'anacode.api.client.AnacodeClient.' + call
    mock.patch(to_mock, empty_json)

    analyzer = client.Analyzer(api, writer, bulk_size=10)
    mocker.spy(analyzer, 'execute_tasks_and_store_output')

    for _ in range(count):
        getattr(analyzer, call)(*args)

    assert analyzer.execute_tasks_and_store_output.call_count == call_count


@pytest.mark.parametrize('call, args', [
    ('scrape', ([], )),
    ('analyze', ([], ['categories'])),
    ('analyze', ([], ['concepts'])),
    ('analyze', ([], ['sentiment'])),
    ('analyze', ([], ['absa'])),
])
def test_analysis_execution(api, mocker, call, args):
    text = ['安全性能很好，很帅气。']
    writer = writers.DataFrameWriter()
    writer.init()

    to_mock = 'anacode.api.client.AnacodeClient.' + call
    mock.patch(to_mock, empty_json)
    mocker.spy(api, call)

    analyzer = client.Analyzer(api, writer, bulk_size=10)
    for _ in range(4):
        getattr(analyzer, call)(*args)

    analyzer.execute_tasks_and_store_output()
    assert getattr(api, call).call_count == 4


def time_consuming(*args, **kwargs):
    time.sleep(0.1)
    return {}


@mock.patch('anacode.api.client.AnacodeClient.analyze', time_consuming)
def test_parallel_queries(api, mocker):
    text = ['安全性能很好，很帅气。']
    writer = writers.DataFrameWriter()
    writer.init()

    mocker.spy(api, 'analyze')
    analyzer = client.Analyzer(api, writer, threads=4, bulk_size=4)

    start = time.time()
    with analyzer:
        for _ in range(4):
            analyzer.analyze(text, ['categories'])
    stop = time.time()
    duration = stop - start
    assert abs(duration - 0.1) < 0.1
