import pytest
from django.urls import reverse, resolve

pytestmark = pytest.mark.django_db


def test_index():
    assert reverse("sample_search:sample_search") == "/sample_search/"
    assert resolve("/sample_search/").view_name == "sample_search:sample_search"
