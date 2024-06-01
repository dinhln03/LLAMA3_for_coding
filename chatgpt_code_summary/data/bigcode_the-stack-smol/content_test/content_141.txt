# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
# Core libs
from typing import TYPE_CHECKING

# Third party libs
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView

# Project libs
from apps.users.models import ClientAddress
from apps.users.serializers.client_address import (
    ClientAddressCreateSerializer,
    ClientAddressRetrieveSerializer,
)

# If type checking, __all__
if TYPE_CHECKING:
    pass

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------


class ClientAddressCreateListView(ListCreateAPIView):
    queryset = ClientAddress.objects.all()
    serializer_class = ClientAddressCreateSerializer


class ClientAddressRetrieveUpdateView(RetrieveUpdateDestroyAPIView):
    queryset = ClientAddress.objects.all()
    serializer_class = ClientAddressRetrieveSerializer
