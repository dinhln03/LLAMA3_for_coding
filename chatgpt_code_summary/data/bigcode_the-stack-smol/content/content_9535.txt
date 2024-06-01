import pytest

from app.models.user import User
from datetime import date


@pytest.fixture(scope='module')
def new_user():
    user = User('test', date(day=1, month=12, year=1989))
    return user


def test_new_user(new_user):
    """
    GIVEN a User model
    WHEN a new User is created
    THEN check the username and birthday fields are defined correctly
    """
    assert new_user.name == 'test'
    assert new_user.birthday == date(day=1, month=12, year=1989)
