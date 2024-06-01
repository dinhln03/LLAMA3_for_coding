from django.test import TestCase
from django.contrib.auth import get_user_model
from core import models


def sample_user(email='rg171195@gmail.com', password='testpass'):
    '''Creating sample user'''
    return get_user_model().objects.create_user(email, password)


class ModelTests(TestCase):

    def test_create_user_with_email_successful(self):
        """Test creating a new user with an email is successful"""
        email = 'rg171195@gmail.com'
        password = 'Password123'
        user = get_user_model().objects.create_user(
            email=email,
            password=password
            )
        self.assertEqual(user.email, email)
        self.assertTrue(user.check_password(password))

    def test_email_normalize(self):
        """Testing weather email is in normalize form or not"""
        email = "test@XYZ.com"
        user = get_user_model().objects.create_user(email, "test123")
        self.assertEqual(user.email, email.lower())

    def test_email_validation(self):
        with self.assertRaises(ValueError):
            get_user_model().objects.create_user(None, 'test123')

    def test_create_superuser(self):
        """Test for creating super user"""
        email = 'rg171195@gmail.com'
        password = 'Password123'
        user = get_user_model().objects.create_superuser(
            email=email,
            password=password
            )

        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    def test_tag_str(self):
        tag = models.Tag.objects.create(user=sample_user(), name='vegan')
        self.assertEqual(str(tag), tag.name)
