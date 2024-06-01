from django.test import TestCase
from django.contrib.auth import get_user_model


class ModelTest(TestCase):

    def test_create_user_with_email_successful(self):
        """이메일로 유저 생성을 성공하는 테스트"""
        email = 'test@testemail.com'
        password = 'testpassword'
        user = get_user_model().objects.create_user(
            email=email,
            password=password
        )

        self.assertEqual(user.email, email)
        self.assertTrue(user.check_password(password))

    def test_new_user_email_normalized(self):
        """이메일이 표준 형식으로 들어오는 테스트"""
        email = 'test@TESTEMAIL.COM'
        user = get_user_model().objects.create_user(email, 'testpw123')

        self.assertEqual(user.email, email.lower())
    
    def test_new_user_missing_email(self):
        """이메일이 입력되지 않았을 때 에러가 발생하는 테스트"""
        with self.assertRaises(ValueError):
            get_user_model().objects.create_user(None, 'testpw123')

    def test_create_new_superuser(self):
        """Superuser를 생성하는 테스트"""
        user = get_user_model().objects.create_superuser(
            'testsuperuser@admin.com',
            'testpw123'
        )

        self.assertTrue(user.is_superuser)
        self.assertTrue(user.is_staff)
