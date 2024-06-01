import models
import serializers
from rest_framework import viewsets, permissions


class apiViewSet(viewsets.ModelViewSet):
    """ViewSet for the api class"""

    queryset = models.api.objects.all()
    serializer_class = serializers.apiSerializer
    permission_classes = [permissions.IsAuthenticated]


