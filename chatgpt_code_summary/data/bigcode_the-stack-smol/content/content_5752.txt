from django.test import TestCase
from django.contrib.auth import get_user_model


class ModelTests(TestCase):
    """ Test creating a new user with an email is successful """
    def test_create_user_with_email_successful(self):
        payload = {'email': 'pudgeinvonyx@gmail.com', 'password': '1111qqqq='}
        user = get_user_model().objects.create_user(
            email=payload['email'],
            password=payload['password']
        )

        self.assertEqual(user.email, payload['email'])
        self.assertTrue(user.check_password(payload['password']))

    def test_create_user_with_lowercase_email(self):
        """ Test creating a new user with an lowercase email words """
        payload = {'email': 'pudgeinvonyx@GMAIL.com', 'password': '1111qqqq='}
        user = get_user_model().objects.create_user(
            email=payload['email'],
            password=payload['password']
        )

        self.assertEqual(user.email, payload['email'].lower())

    def test_create_user_with_invalid_email(self):
        """ Test creating a new user with an invalid email address """
        with self.assertRaises(ValueError):
            get_user_model().objects.create_user(None, "1234325")

    def test_create_superuser_is_successful(self):
        """ Test that create a new superuser """
        user = get_user_model().objects.create_superuser("pudge@com.com", '1234')

        self.assertTrue(user.is_superuser)
        self.assertTrue(user.is_staff)


