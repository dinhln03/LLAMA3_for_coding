from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import TestCase

from rest_framework import status
from rest_framework.test import APIClient

from core.models import Ingredient

from recipe_app.serializers import IngredientSerializer

INGREDIENTS_URL = reverse('recipe_app:ingredient-list')


def create_user(**params):
    return get_user_model().objects.create_user(**params)


class PublicIngredientsAPITests(TestCase):
    """Test endpoints that don't require authentication."""

    def setUp(self):
        self.client = APIClient()

    def test_login_required_to_view_ingredients(self):
        """Test that authentication is needed to view the ingredients."""

        res = self.client.get(INGREDIENTS_URL)
        self.assertEqual(res.status_code, status.HTTP_401_UNAUTHORIZED)


class PrivateIngredientsAPITests(TestCase):
    """Test endpoints that require authentication."""

    def setUp(self):
        self.client = APIClient()

        self.user = create_user(
            fname='Test',
            lname='User',
            email='test@gmail.com',
            password='testpass'
        )
        self.client.force_authenticate(user=self.user)

    def test_retrieve_ingredients_is_successful(self):
        """Test retrieve ingredients"""

        Ingredient.objects.create(user=self.user, name='Carrot')
        Ingredient.objects.create(user=self.user, name='Lemon')

        res = self.client.get(INGREDIENTS_URL)

        ingredients = Ingredient.objects.all().order_by('-name')
        serializer = IngredientSerializer(ingredients, many=True)

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(res.data, serializer.data)

    def test_retrieved_ingredients_limited_to_user(self):
        """Tests that only the user's ingredients are retrieved"""

        user2 = create_user(
            fname='Test2',
            lname='User2',
            email='test2@gmail.com',
            password='test2pass'
        )

        Ingredient.objects.create(user=user2, name='Carrot')
        ingredient = Ingredient.objects.create(user=self.user, name='Lemon')

        res = self.client.get(INGREDIENTS_URL)

        self.assertEqual(res.status_code, status.HTTP_200_OK)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0]['name'], ingredient.name)

    def test_create_ingredient_is_successful(self):
        """Test that creating a new ingredient is successful."""

        payload = {
            'name': 'Lemon'
        }

        self.client.post(INGREDIENTS_URL, payload)

        exists = Ingredient.objects.filter(
            user=self.user,
            name=payload['name']
        ).exists()

        self.assertTrue(exists)

    def test_create_ingredient_with_invalid_details_invalid(self):
        """Test that ingredients is not created with invalid details"""

        payload = {
            'name': ''
        }

        res = self.client.post(INGREDIENTS_URL, payload)

        self.assertEqual(res.status_code, status.HTTP_400_BAD_REQUEST)
