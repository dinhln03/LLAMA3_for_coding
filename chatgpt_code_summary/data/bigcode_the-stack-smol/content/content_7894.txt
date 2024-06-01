import json
import os

from djoser.conf import settings as djoser_settings
from djoser.compat import get_user_email
from django.utils.timezone import now
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes, action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.db import transaction
from django.conf import settings

from .authentication import WebpageTokenAuth
from .models import AHJUserMaintains, AHJ, User, APIToken, Contact, PreferredContactMethod
from .permissions import IsSuperuser
from .serializers import UserSerializer
from djoser.views import UserViewSet

from .utils import get_enum_value_row, filter_dict_keys, ENUM_FIELDS


@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated])
class ConfirmPasswordReset(UserViewSet):

    @action(["post"], detail=False)
    def reset_password_confirm(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        serializer.user.set_password(serializer.data["new_password"])
        if hasattr(serializer.user, "last_login"):
            serializer.user.last_login = now()
        serializer.user.is_active = True # The purpose of overwriting this endpoint is to set users as active if performing password reset confirm.
        serializer.user.save()           # The user had to access their email account to perform a password reset.

        if djoser_settings.PASSWORD_CHANGED_EMAIL_CONFIRMATION:
            context = {"user": serializer.user}
            to = [get_user_email(serializer.user)]
            djoser_settings.EMAIL.password_changed_confirmation(self.request, context).send(to)
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['GET'])
@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated])
def get_active_user(request):
    """
    Endpoint for getting the active user
    through the authtoken
    """
    return Response(UserSerializer(request.user, context={'is_public_view': False}).data, status=status.HTTP_200_OK)


@api_view(['GET'])
def get_single_user(request, username):
    """
    Function view for getting a single user with the specified Username = username
    """
    context = {'is_public_view': True}
    if request.auth is not None and request.user.Username == username:
        context['is_public_view'] = False
    try:
        user = User.objects.get(Username=username)
        return Response(UserSerializer(user, context=context).data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(str(e), status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated])
def user_update(request):
    """
    Update the user profile associated with the requesting user.
    """
    changeable_user_fields = {'Username', 'PersonalBio', 'URL', 'CompanyAffiliation'}
    changeable_contact_fields = {'FirstName', 'LastName', 'URL', 'WorkPhone', 'PreferredContactMethod', 'Title'}
    user_data = filter_dict_keys(request.data, changeable_user_fields)
    contact_data = filter_dict_keys(request.data, changeable_contact_fields)
    for field in ENUM_FIELDS.intersection(contact_data.keys()):
        contact_data[field] = get_enum_value_row(field, contact_data[field])
    user = request.user
    User.objects.filter(UserID=user.UserID).update(**user_data)
    Contact.objects.filter(ContactID=user.ContactID.ContactID).update(**contact_data)
    return Response('Success', status=status.HTTP_200_OK)


@api_view(['GET'])
@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated, IsSuperuser])
def create_api_token(request):
    try:
        user = request.user
        with transaction.atomic():
            APIToken.objects.filter(user=user).delete()
            api_token = APIToken.objects.create(user=user)
        return Response({'auth_token': api_token.key}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response(str(e), status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated, IsSuperuser])
def set_ahj_maintainer(request):
    """
    View to assign a user as a data maintainer of an AHJ
    Expects a Username and a the primary key of an AHJ (AHJPK)
    """
    try:
        username = request.data['Username']
        ahjpk = request.data['AHJPK']
        user = User.objects.get(Username=username)
        ahj = AHJ.objects.get(AHJPK=ahjpk)
        maintainer_record = AHJUserMaintains.objects.filter(AHJPK=ahj, UserID=user)
        if maintainer_record.exists():
            maintainer_record.update(MaintainerStatus=True)
        else:
            AHJUserMaintains.objects.create(UserID=user, AHJPK=ahj, MaintainerStatus=True)
        return Response(UserSerializer(user).data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(str(e), status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@authentication_classes([WebpageTokenAuth])
@permission_classes([IsAuthenticated, IsSuperuser])
def remove_ahj_maintainer(request):
    """
    View to revoke a user as a data maintainer of an AHJ
    Expects a user's webpage token and a the primary key of an AHJ (AHJPK)
    """
    try:
        username = request.data['Username']
        ahjpk = request.data['AHJPK']
        user = User.objects.get(Username=username)
        ahj = AHJ.objects.get(AHJPK=ahjpk)
        AHJUserMaintains.objects.filter(AHJPK=ahj, UserID=user).update(MaintainerStatus=False)
        return Response(UserSerializer(user).data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
