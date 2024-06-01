from django.test import TestCase
from django.urls import reverse

from rest_framework.test import APIClient
from rest_framework import status

from core.models import Recipe, Ingredient

RECIPE_URL = reverse('recipe:recipe-list')


def recipe_url(id):
    """Construct URL for a single recipe based on its ID"""
    return reverse('recipe:recipe-detail', args=[id])


def create_sample_recipe(**params):
    """Helper function to create a user"""
    return Recipe.objects.create(**params)


class RecipeAPITests(TestCase):

    def setUp(self):
        self.client = APIClient()

    def test_create_recipe_with_ingredients(self):
        """Test creating a recipe including ingredients"""
        payload = {
            'name': 'Vegan Roast Dinner',
            'description': 'Roasted potatoes and mushroom wellington'
                           ' with vegetables and gravy.',
            'ingredients': [
                {'name': 'carrots'},
                {'name': 'potatoes'},
                {'name': 'mushrooms'},
            ]
        }

        response = self.client.post(RECIPE_URL, payload, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            payload['name'],
            Recipe.objects.get(id=response.data['id']).name
        )
        self.assertEquals(
            len(response.data['ingredients']),
            len(payload['ingredients'])
        )

    def test_get_recipes(self):
        """Test retrieving a recipe"""
        create_sample_recipe(
            name='Roast Dinner',
            description='Roasted potatoes and chicken'
                        ' with vegetables and gravy.'
        )

        create_sample_recipe(
            name='Beans on Toast',
            description='Just the best.'
        )

        response = self.client.get(RECIPE_URL)

        recipes = Recipe.objects.all().order_by('-name')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), len(recipes))

    def test_get_recipe(self):
        """Test retrieving a single recipe using name as filter"""
        test_recipe_name = 'Beans on Toast'

        create_sample_recipe(
            name='Roast Dinner',
            description='Roasted potatoes and chicken'
                        ' with vegetables and gravy.'
        )

        create_sample_recipe(
            name=test_recipe_name,
            description='Just the best recipe.'
        )

        response = self.client.get(RECIPE_URL, {'name': test_recipe_name})

        recipes = Recipe.objects.all().order_by('-name')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertNotEqual(len(response.data), len(recipes))
        self.assertEqual(response.data[0]['name'], test_recipe_name)

    def test_update_recipe(self):
        """Test updating a recipe"""
        self.recipe = create_sample_recipe(
            name='Roast Dinner',
            description='Roasted potatoes and chicken'
                        ' with vegetables and gravy.'
        )

        payload = {
            'name': 'Vegan Roast Dinner',
            'description': 'Roasted potatoes and mushroom wellington'
                           ' with vegetables and gravy.'
        }

        response = self.client.patch(
            recipe_url(self.recipe.id),
            payload, format='json'
        )

        self.recipe.refresh_from_db()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self.recipe.name, response.data['name'])
        self.assertEqual(self.recipe.description, response.data['description'])

    def test_delete_recipe(self):
        """Test deleting a recipe"""
        self.recipe = create_sample_recipe(
            name='Carrot Cake',
            description='Sponge cake with hella carrots.'
        )

        response = self.client.delete(
            recipe_url(self.recipe.id),
            format='json'
        )

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Recipe.objects.all())

    def test_get_recipes_with_ingredients(self):
        """Test retrieving a recipe including ingredients"""
        self.recipe = create_sample_recipe(
            name='Carrot Cake',
            description='Sponge cake with hella carrots.'
        )

        Ingredient.objects.create(name='Carrots', recipe=self.recipe)
        Ingredient.objects.create(name='Icing Sugar', recipe=self.recipe)

        response = self.client.get(RECIPE_URL)

        ingredients = Ingredient.objects.all()

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEquals(
            len(response.data[0]['ingredients']),
            len(ingredients)
        )

    def test_update_recipe_ingredients(self):
        """Test updating a recipe with ingredients included"""
        self.recipe = create_sample_recipe(
            name='Roast Dinner',
            description='Roasted potatoes and chicken'
                        ' with vegetables and gravy.'
        )

        payload = {
            'name': 'Vegan Roast Dinner',
            'description': 'Roasted potatoes and mushroom wellington'
                           ' with vegetables and gravy.',
            'ingredients': [
                {'name': 'carrots'},
                {'name': 'potatoes'},
                {'name': 'mushrooms'},
            ]
        }

        response = self.client.patch(
            recipe_url(self.recipe.id),
            payload, format='json'
        )

        self.recipe.refresh_from_db()

        ingredients = Ingredient.objects.all()

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(ingredients), len(payload['ingredients']))
        self.assertEqual(ingredients[0].recipe.name, payload['name'])

    def test_delete_recipe_with_ingredients(self):
        """Test deleting a recipe with ingredients included"""
        self.recipe = create_sample_recipe(
            name='Carrot Cake',
            description='Sponge cake with hella carrots.'
        )

        Ingredient.objects.create(name='Carrots', recipe=self.recipe)
        Ingredient.objects.create(name='Icing Sugar', recipe=self.recipe)

        response = self.client.delete(
            recipe_url(self.recipe.id),
            format='json'
        )

        ingredients = Ingredient.objects.all()

        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertFalse(Recipe.objects.all())
        self.assertFalse(len(ingredients), 0)
