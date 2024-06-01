"""
"""

import unittest
from unittest.mock import Mock, patch

from wheezy.core import __version__, httpclient
from wheezy.core.gzip import compress


class HTTPClientTestCase(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.object(httpclient, "HTTPConnection")
        self.mock_c_class = self.patcher.start()
        self.headers = [("date", "Sat, 12 Oct 2013 18:29:13 GMT")]
        self.mock_response = Mock()
        self.mock_response.getheaders.return_value = self.headers
        self.mock_response.read.return_value = "".encode("utf-8")
        self.mock_c = Mock()
        self.mock_c.getresponse.return_value = self.mock_response
        self.mock_c_class.return_value = self.mock_c
        self.client = httpclient.HTTPClient(
            "http://localhost:8080/api/v1/",
            headers={"User-Agent": "wheezy/%s" % __version__},
        )

    def tearDown(self):
        self.patcher.stop()

    def test_init(self):
        self.mock_c_class.assert_called_once_with("localhost:8080")
        assert "/api/v1/" == self.client.path
        assert {} == self.client.cookies
        assert self.client.headers is None

    def test_get(self):
        self.mock_response.status = 200
        assert 200 == self.client.get("auth/token")
        assert self.mock_c.connect.called
        assert self.mock_c.request.called
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "GET" == method
        assert "/api/v1/auth/token" == path
        assert "" == body
        assert self.client.default_headers == headers
        assert "gzip" == headers["Accept-Encoding"]
        assert "close" == headers["Connection"]
        assert 3 == len(headers)

    def test_ajax_get(self):
        self.client.ajax_get("auth/token")
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "XMLHttpRequest" == headers["X-Requested-With"]

    def test_get_query(self):
        self.client.get("auth/token", params={"a": ["1"]})
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "/api/v1/auth/token?a=1" == path

    def test_head(self):
        self.client.head("auth/token")
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "HEAD" == method

    def test_post(self):
        self.client.post(
            "auth/token",
            params={
                "a": ["1"],
            },
        )
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "POST" == method
        assert "/api/v1/auth/token" == path
        assert "a=1" == body
        assert "application/x-www-form-urlencoded" == headers["Content-Type"]

    def test_ajax_post(self):
        self.client.ajax_post("auth/token", params={"a": ["1"]})
        assert self.mock_c.request.called
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "XMLHttpRequest" == headers["X-Requested-With"]

    def test_post_content(self):
        self.client.ajax_post(
            "auth/token", content_type="application/json", body='{"a":1}'
        )
        assert self.mock_c.request.called
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "application/json" == headers["Content-Type"]
        assert '{"a":1}' == body

    def test_follow(self):
        self.mock_response.status = 303
        self.headers.append(("location", "http://localhost:8080/error/401"))
        assert 303 == self.client.get("auth/token")
        self.client.follow()
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "GET" == method
        assert "/error/401" == path

    def test_cookies(self):
        self.headers.append(("set-cookie", "_x=1; path=/; httponly"))
        self.client.get("auth/token")
        assert self.client.cookies
        assert "1" == self.client.cookies["_x"]

        self.headers.append(("set-cookie", "_x=; path=/; httponly"))
        self.client.get("auth/token")
        assert not self.client.cookies

    def test_assert_json(self):
        """Expecting json response but content type is not valid."""
        self.headers.append(("content-type", "text/html; charset=UTF-8"))
        self.client.get("auth/token")
        self.assertRaises(AssertionError, lambda: self.client.json)

    def test_json(self):
        """json response."""
        patcher = patch.object(httpclient, "json_loads")
        mock_json_loads = patcher.start()
        mock_json_loads.return_value = {}
        self.headers.append(
            ("content-type", "application/json; charset=UTF-8")
        )
        self.mock_response.read.return_value = "{}".encode("utf-8")
        self.client.get("auth/token")
        assert {} == self.client.json
        patcher.stop()

    def test_gzip(self):
        """Ensure gzip decompression."""
        self.headers.append(("content-encoding", "gzip"))
        self.mock_response.read.return_value = compress("test".encode("utf-8"))
        self.client.get("auth/token")
        assert "test" == self.client.content

    def test_etag(self):
        """ETag processing."""
        self.headers.append(("etag", '"ca231fbc"'))
        self.client.get("auth/token")
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert "If-None-Match" not in headers
        assert '"ca231fbc"' == self.client.etags["/api/v1/auth/token"]
        self.client.get("auth/token")
        method, path, body, headers = self.mock_c.request.call_args[0]
        assert '"ca231fbc"' == headers["If-None-Match"]
