import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def db_user():
    user_data = {
        'email': 'test@example.com',
        'password': 'testpass123',
        'first_name': 'Jack',
        'last_name': 'Programmer',
    }
    return get_user_model().objects.create_user(**user_data)
