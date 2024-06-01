from rest_framework import serializers
from .models import Canteen
from accounts.serializers import UserForSerializer
from model_location.serializers import CityViewSerializer
from model_media.serializers import MediaViewSerializer


# Canteen model serializer
class CanteenSerializer(serializers.ModelSerializer):
    class Meta:
        model = Canteen
        fields = '__all__'


# Canteen model serializer to view
class CanteenViewSerializer(serializers.ModelSerializer):
    user = UserForSerializer(read_only=True)
    city = CityViewSerializer(read_only=True)
    images = MediaViewSerializer(read_only=True, many=True)

    class Meta:
        model = Canteen
        fields = '__all__'
