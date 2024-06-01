import logging

from django.utils.decorators import method_decorator
from django_filters.rest_framework import DjangoFilterBackend
from drf_yasg.utils import swagger_auto_schema
from rest_condition import Or
from rest_framework import filters, viewsets
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import AllowAny, IsAuthenticated

from base.documentation import jwt_header
from users.models import Device
from users.serializers import DeviceSerializer, DeviceListSerializer

logger = logging.getLogger('user_app')


@method_decorator(name='list', decorator=swagger_auto_schema(manual_parameters=[jwt_header]))
@method_decorator(name='retrieve', decorator=swagger_auto_schema(manual_parameters=[jwt_header]))
@method_decorator(name='create', decorator=swagger_auto_schema(manual_parameters=[jwt_header]))
class DeviceViewSet(viewsets.ModelViewSet):
    """
    Creating Device
    """
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    queryset = Device.objects.all()
    serializer_class = DeviceSerializer

    filter_backends = [filters.SearchFilter, DjangoFilterBackend]
    search_fields = ['user_id', 'user__username']
    filter_fields = ['user_id', 'user__username']
    http_method_names = ['get', 'post']

    def get_queryset(self):
        if IsAdminUser.has_permission(self, request=self.request, view=self.get_view_name()):
            return super().get_queryset()
        return super().get_queryset().filter(user=self.request.user)

    # def get_permissions(self):
    #     return [permission() for permission in permission_classes]

    def get_serializer_class(self):
        if self.action in ['list']:
            return DeviceListSerializer
        return DeviceSerializer


class DeviceList(APIView):
    """
    An API endpoint for Getting device list
    """
    permission_classes = [AllowAny]

    def get(self, request, user_id):
        queryset = Device.objects.filter(user_id=user_id).select_related('user')
        serializer_context = {
            'request': request
        }
        serializer = DeviceListSerializer(queryset, context=serializer_context, many=True)
        data = serializer.data
        if data:
            return Response(data)
        return Response({'message': f"No Device data found", 'data': []})