from shoppinglist.models import Ingredient
from rest_framework import serializers

class IngredientSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Ingredient
        fields = ('account', 'member', 'ref_date', 'ref_meal',
                  'ingredient', 'created', 'ingredient_there')
