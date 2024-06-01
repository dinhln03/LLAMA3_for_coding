import datetime
import logging
import os

import elastalert.elastalert
import elastalert.utils.util
import mock
import pytest
from elastalert import config
from elastalert.ruletypes import AnyRule
from elastalert.utils.time import dt_to_ts, ts_to_dt

writeback_index = "wb"


def pytest_addoption(parser):
    parser.addoption(
        "--runelasticsearch",
        action="store_true",
        default=False,
        help="run elasticsearch tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runelasticsearch"):
        # --runelasticsearch given in cli: run elasticsearch tests, skip ordinary unit
        # tests
        skip_unit_tests = pytest.mark.skip(
            reason="not running when --runelasticsearch option is used to run"
        )
        for item in items:
            if "elasticsearch" not in item.keywords:
                item.add_marker(skip_unit_tests)
    else:
        # skip elasticsearch tests
        skip_elasticsearch = pytest.mark.skip(
            reason="need --runelasticsearch option to run"
        )
        for item in items:
            if "elasticsearch" in item.keywords:
                item.add_marker(skip_elasticsearch)


@pytest.fixture
def cls_monkeypatch(request, monkeypatch):
    request.cls.monkeypatch = monkeypatch


@pytest.fixture(scope="function", autouse=True)
def reset_loggers():
    """Prevent logging handlers from capturing temporary file handles.

    For example, a test that uses the `capsys` fixture and calls
    `logging.exception()` will initialize logging with a default handler that
    captures `sys.stderr`.  When the test ends, the file handles will be closed
    and `sys.stderr` will be returned to its original handle, but the logging
    will have a dangling reference to the temporary handle used in the `capsys`
    fixture.

    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)


class mock_es_indices_client(object):
    def __init__(self):
        self.exists = mock.Mock(return_value=True)


class mock_es_client(object):
    def __init__(self, host="es", port=14900):
        self.host = host
        self.port = port
        self.return_hits = []
        self.search = mock.Mock()
        self.deprecated_search = mock.Mock()
        self.create = mock.Mock()
        self.index = mock.Mock()
        self.delete = mock.Mock()
        self.info = mock.Mock(
            return_value={"status": 200, "name": "foo", "version": {"number": "2.0"}}
        )
        self.ping = mock.Mock(return_value=True)
        self.indices = mock_es_indices_client()
        self.es_version = mock.Mock(return_value="2.0")
        self.is_atleastfive = mock.Mock(return_value=False)
        self.is_atleastsix = mock.Mock(return_value=False)
        self.is_atleastsixtwo = mock.Mock(return_value=False)
        self.is_atleastsixsix = mock.Mock(return_value=False)
        self.is_atleastseven = mock.Mock(return_value=False)

        def writeback_index_side_effect(index, doc_type):
            if doc_type == "silence":
                return index + "_silence"
            elif doc_type == "past_elastalert":
                return index + "_past"
            elif doc_type == "elastalert_status":
                return index + "_status"
            elif doc_type == "elastalert_error":
                return index + "_error"
            return index

        self.resolve_writeback_index = mock.Mock(
            side_effect=writeback_index_side_effect
        )


def mock_ruletype(conf, es):
    rule = AnyRule(conf, es=es)
    rule.add_data = mock.Mock()
    rule.add_count_data = mock.Mock()
    rule.garbage_collect = mock.Mock()
    rule.add_terms_data = mock.Mock()
    rule.find_pending_aggregate_alert = mock.Mock()
    rule.find_pending_aggregate_alert.return_value = False
    rule.is_silenced = mock.Mock()
    rule.is_silenced.return_value = False
    rule.matches = []
    rule.get_match_data = lambda x: x
    rule.get_match_str = lambda x: "some stuff happened"
    rule.garbage_collect = mock.Mock()
    return rule


class mock_alert(object):
    def __init__(self):
        self.alert = mock.Mock()

    def get_info(self):
        return {"type": "mock"}


@pytest.fixture
def configured(monkeypatch):

    test_args = mock.Mock()
    test_args.config = "test_config"
    test_args.rule = None
    test_args.debug = False
    test_args.es_debug_trace = None
    test_args.silence = False
    test_args.timeout = 0

    _conf = {
        "args": test_args,
        "debug": False,
        "rules_loader": "test",
        "rules_folder": "rules",
        "run_every": datetime.timedelta(minutes=10),
        "buffer_time": datetime.timedelta(minutes=5),
        "alert_time_limit": datetime.timedelta(hours=24),
        "es_client": config.ESClient(
            es_host="es",
            es_port=12345,
            es_password="",
            es_username="",
            es_conn_timeout=1234,
            es_url_prefix="es/test",
            es_send_get_body_as="GET",
        ),
        "writeback_index": "wb",
        "writeback_alias": "wb_a",
        "max_query_size": 10000,
        "old_query_limit": datetime.timedelta(weeks=1),
        "disable_rules_on_error": False,
        "scroll_keepalive": "30s",
    }

    monkeypatch.setattr(config, "_cfg", config.Config(**_conf))


@pytest.fixture
def ea():
    test_args = mock.Mock()
    test_args.config = "test_config"
    test_args.rule = None
    test_args.debug = False
    test_args.es_debug_trace = None
    test_args.silence = False
    test_args.timeout = datetime.timedelta(seconds=0)
    test_args.end = None

    _conf = {
        "args": test_args,
        "debug": False,
        "rules_loader": "test",
        "rules_folder": "rules",
        "run_every": datetime.timedelta(minutes=10),
        "buffer_time": datetime.timedelta(minutes=5),
        "alert_time_limit": datetime.timedelta(hours=24),
        "es_client": config.ESClient(
            es_host="es",
            es_port=12345,
            es_password="",
            es_username="",
            es_conn_timeout=1234,
            es_url_prefix="es/test",
            es_send_get_body_as="GET",
        ),
        "mail_settings": config.MailSettings(notify_email=[]),
        "writeback_index": "wb",
        "writeback_alias": "wb_a",
        "max_query_size": 10000,
        "old_query_limit": datetime.timedelta(weeks=1),
        "disable_rules_on_error": False,
        "scroll_keepalive": "30s",
    }

    conf = config.Config(**_conf)

    rules = {
        "testrule": {
            "name": "testrule",
            "es_host": "",
            "es_port": 14900,
            "index": "idx",
            "filter": [],
            "include": ["@timestamp"],
            "aggregation": datetime.timedelta(0),
            "realert": datetime.timedelta(0),
            "processed_hits": {},
            "timestamp_field": "@timestamp",
            "match_enhancements": [],
            "rule_file": "blah.yaml",
            "max_query_size": 10000,
            "ts_to_dt": ts_to_dt,
            "dt_to_ts": dt_to_ts,
            "_source_enabled": True,
            "run_every": datetime.timedelta(seconds=15),
        }
    }
    elastalert.elastalert.elasticsearch_client = mock_es_client

    class mock_rule_loader(object):
        required_globals = frozenset([])

        def __init__(self, conf):
            self.base_config = conf
            self.load_configuration = mock.Mock()

        def load(self, args):
            return rules

        def get_hashes(self, args):
            return {}

        def load_rule(self, str: str):
            return {}

    with mock.patch("elastalert.elastalert.BackgroundScheduler"):
        with mock.patch(
            "elastalert.elastalert.config.Config.load_config"
        ) as load_config:
            with mock.patch(
                "elastalert.elastalert.loader_mapping"
            ) as loader_mapping, mock.patch(
                "elastalert.elastalert.config.configure_logging"
            ):
                loader_mapping.get.return_value = mock_rule_loader
                load_config.return_value = conf
                ea = elastalert.elastalert.ElastAlerter(["--pin_rules"])
    rules["testrule"]["alert"] = [mock_alert()]
    ea.rule_es = mock_es_client()
    ea.rule_es.is_atleastsixtwo.return_value = True
    ea.rule_es.is_atleastfive.return_value = True
    ea.rule_es.index.return_value = {"_id": "ABCD", "created": True}
    ea.rules["testrule"]["type"] = mock_ruletype(rules["testrule"], ea.rule_es)
    ea.testrule = ea.rules["testrule"]["type"]
    ea.conf = conf

    ea.writeback_es = mock_es_client()
    ea.writeback_es.is_atleastsixtwo.return_value = True
    ea.writeback_es.is_atleastfive.return_value = True
    ea.writeback_es.search.return_value = {
        "hits": {"total": {"value": "0"}, "hits": []}
    }
    ea.writeback_es.deprecated_search.return_value = {"hits": {"hits": []}}
    ea.writeback_es.index.return_value = {"_id": "ABCD", "created": True}
    ea.es = mock_es_client()
    ea.es.index.return_value = {"_id": "ABCD", "created": True}
    ea.thread_data.num_hits = 0
    ea.thread_data.num_dupes = 0
    return ea


@pytest.fixture(scope="function")
def environ():
    """py.test fixture to get a fresh mutable environment."""
    old_env = os.environ
    new_env = dict(list(old_env.items()))
    os.environ = new_env
    yield os.environ
    os.environ = old_env
