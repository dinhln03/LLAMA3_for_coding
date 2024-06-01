from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status


CREATE_USER_URL = reverse('user:create')
TOKEN_URL = reverse('user:token')


def create_user(**params):
    return get_user_model().objects.create_user(**params)


class PublicUserApiTests(TestCase):
    """The the user API (public)"""

    def setUp(self):
        self.client = APIClient()

    def test_create_valid_user_successful(self):
        """Test creating user with valid payload is successful"""
        payload = {
            'email': 'john@doe.com',
            'password': 'testpass',
            'name': 'John Doe'
        }

        res = self.client.post(CREATE_USER_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_201_CREATED)
        user = get_user_model().objects.get(**res.data)
        self.assertTrue(user.check_password(payload['password']))
        self.assertNotIn('password', res.data)

    def test_user_exists(self):
        payload = {
            'email': 'john@doe.com',
            'password': 'testpass',
            "name": 'John Doe'
        }
        create_user(**payload)
        res = self.client.post(CREATE_USER_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)

    def test_password_too_short(self):
        """tests that the password must be more than 5 characters"""
        payload = {
            'email': 'john@doe.com',
            'password': 'pass',
            "name": 'John Doe'
        }
        res = self.client.post(CREATE_USER_URL, payload)
        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
        user_exists = get_user_model().objects.filter(
            email=payload['email']
        ).exists()
        self.assertFalse(user_exists)

    def test_create_token_for_user(self):
        """Test that a token is created for a user"""
        payload = {'email': 'test@django.io', 'password': 'testpass'}
        create_user(**payload)
        res = self.client.post(TOKEN_URL, payload)
        self.assertTrue(res.status_code, status.HTTP_200_OK)
        self.assertIn('token', res.data)

    def test_create_token_invalid_credentials(self):
        """Test that token is not created if invalid credentials are given"""
        create_user(email='test@django.com', password='testpass')
        payload = {'email': 'test@django.com', 'password': 'wrong'}
        res = self.client.post(TOKEN_URL, payload)
        self.assertTrue(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn('token', res.data)

    def test_create_token_no_user(self):
        """Test that token is not created if user does not exist"""
        payload = {'email': 'test@django.com', 'password': 'wrong'}
        res = self.client.post(TOKEN_URL, payload)
        self.assertTrue(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn('token', res.data)

    def test_create_token_no_missing_field(self):
        """Test that token is not created if email/password not given"""
        res = self.client.post(
                                TOKEN_URL,
                                {'email': 'test@django.com', 'password': ''})
        self.assertTrue(res.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertNotIn('token', res.data)
