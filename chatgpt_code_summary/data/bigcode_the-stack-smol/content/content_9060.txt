from django.contrib.auth import get_user_model, authenticate
from rest_framework import serializers
from django.utils.translation import ugettext_lazy as _


class UsersSerializer(serializers.ModelSerializer):
    """Serializer for users object"""

    class Meta:
        model = get_user_model()
        fields = ('email', 'password', 'name', 'cf_handle', 'cf_pass',\
                  'uri_handle', 'uri_pass', 'uva_handle', 'uva_pass')
        extra_kwargs = {
            'password': {
                'write_only': True,
                'min_length': 5,
                'style': {'input_type': 'password'}
            },
            'uva_pass': {
                'write_only': True,
                'style': {'input_type': 'password'}
            },
            'uri_pass': {
                'write_only': True,
                'style': {'input_type': 'password'}
            }
        }

    def create(self, validated_data):
        """Create a new user with validated password and return it"""
        return get_user_model().objects.create_user(**validated_data)

    def update(self, instance, validated_data):
        """update user data with encrypted password"""
        password = validated_data.pop('password', None)
        user = super().update(instance, validated_data)
        if password:
            user.set_password(password)
            user.save()
        return user


class AuthTokenSerializer(serializers.Serializer):
    """Serializer for the user authentication object"""
    email = serializers.CharField()
    password = serializers.CharField(
        style={'input_type': 'password'},
        trim_whitespace=False
    )

    def validate(self, attrs):
        """validate and authenticate the user"""
        email = attrs.get('email')
        password = attrs.get('password')

        user = authenticate(
            request=self.context.get('request'),
            username=email,
            password=password
        )
        if not user:
            msg = _('Unable to authenticate with the provided credentials')
            raise serializers.ValidationError(msg, code='authentication')

        attrs['user'] = user
        return attrs
