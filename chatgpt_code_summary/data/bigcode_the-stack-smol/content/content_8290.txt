# Future
from __future__ import division, print_function, unicode_literals

# Standard Library
import time

# Third Party
import pytest
import ratelimit

# DocumentCloud
from documentcloud.constants import RATE_LIMIT
from documentcloud.exceptions import APIError, CredentialsFailedError

# pylint: disable=protected-access


def test_set_tokens_credentials(client):
    """Test setting the tokens using credentials"""
    client.refresh_token = None
    del client.session.headers["Authorization"]
    client._set_tokens()
    assert client.refresh_token
    assert "Authorization" in client.session.headers


def test_set_tokens_refresh(client):
    """Test setting the tokens using refresh token"""
    # first set tokens sets, refresh token, second one uses it
    client.refresh_token = None
    del client.session.headers["Authorization"]
    client._set_tokens()
    client._set_tokens()
    assert client.refresh_token
    assert "Authorization" in client.session.headers


def test_set_tokens_none(public_client):
    """Test setting the tokens with no credentials"""
    public_client._set_tokens()
    assert public_client.refresh_token is None
    assert "Authorization" not in public_client.session.headers


def test_get_tokens(client):
    """Test getting access and refresh tokens using valid credentials"""
    access, refresh = client._get_tokens(client.username, client.password)
    assert access
    assert refresh


def test_get_tokens_bad_credentials(client):
    """Test getting access and refresh tokens using invalid credentials"""
    with pytest.raises(CredentialsFailedError):
        client._get_tokens(client.username, "foo")


def test_refresh_tokens(client):
    """Test refreshing the tokens"""
    access, refresh = client._refresh_tokens(client.refresh_token)
    assert access
    assert refresh


def test_user_id(client):
    assert client.user_id


def test_user_id_public(public_client):
    # pylint: disable=pointless-statement
    with pytest.raises(APIError, match=r"404"):
        public_client.user_id


def test_bad_attr(client):
    with pytest.raises(AttributeError):
        assert client.foo


def test_rate_limit(rate_client):
    with pytest.raises(ratelimit.RateLimitException):
        for _ in range(RATE_LIMIT * 2):
            rate_client.users.get("me")


@pytest.mark.short
@pytest.mark.vcr(cassette_library_dir="tests/cassettes/short_fixtures")
def test_expired_access_token(short_client, record_mode):
    # get fresh tokens
    short_client._set_tokens()
    old_refresh_token = short_client.refresh_token
    # wait for the access token to expire
    if record_mode == "all":
        time.sleep(3)
    # make a request
    assert short_client.users.get("me")
    # check the refresh token was updated
    assert old_refresh_token != short_client.refresh_token


@pytest.mark.short
@pytest.mark.vcr(cassette_library_dir="tests/cassettes/short_fixtures")
def test_expired_refresh_token(short_client, record_mode):
    # get fresh tokens
    short_client._set_tokens()
    old_refresh_token = short_client.refresh_token
    # wait for the access and refresh tokens to expire
    if record_mode == "all":
        time.sleep(6)
    # make a request
    assert short_client.users.get("me")
    # check the refresh token was updated
    assert old_refresh_token != short_client.refresh_token
