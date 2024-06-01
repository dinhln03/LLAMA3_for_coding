from rest_framework import serializers
from django.contrib.auth.models import User
from .models import *



class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model= User
        fields = ['username','first_name','last_name','password','email','is_superuser']

class RolesSerializer(serializers.ModelSerializer):
    class Meta:
        model= Roles
        fields = ['name']
