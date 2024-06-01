from django.test import TestCase

from authors.apps.authentication.models import User


class UserModelTest(TestCase):
    """
    Test Suite for the User model class, User authentication.
    """

    def test_create_user(self):
        """
        Test User model can create a user successfully
        """
        self.assertIsInstance(
            User.objects.create_user(username="username",
                                     email="username@mail.com",
                                     password="password"), User)
