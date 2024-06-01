from django_filters.rest_framework import DjangoFilterBackend, FilterSet
from rest_framework import generics, viewsets, permissions
from rest_framework import filters
from drf_haystack.viewsets import HaystackViewSet
from drf_haystack.filters import HaystackAutocompleteFilter
from drf_haystack.serializers import HaystackSerializer

from patents.models import Patent
from .serializers import PatentSerializer, PatentDetailSerializer, \
    PatentIndexSerializer, AutocompleteSerializer


class PatentListViewSet(viewsets.ModelViewSet):
    queryset = Patent.objects.all().order_by('publication_number')
    serializer_class = PatentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.SearchFilter]
    search_fields = ['title', 'abstract', 'claims']


class PatentListAPIView(generics.ListAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentSerializer
    permission_classes = [permissions.IsAuthenticated]


class PatentDetailAPIView(generics.RetrieveAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentDetailSerializer
    permission_classes = [permissions.IsAuthenticated]


class PatentCreateAPIView(generics.CreateAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentDetailSerializer
    permission_classes = [permissions.IsAuthenticated]


class PatentUpdateAPIView(generics.UpdateAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentDetailSerializer
    permission_classes = [permissions.IsAuthenticated]


class PatentDeleteAPIView(generics.DestroyAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentSerializer
    permission_classes = [permissions.IsAuthenticated]


class PatentFilter(FilterSet):
    class Meta:
        model = Patent
        fields = ['publication_number', 'title', 'abstract', 'claims']


class PatentListFilterAPIView(generics.ListAPIView):
    queryset = Patent.objects.all()
    serializer_class = PatentSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = PatentFilter
    permission_classes = [permissions.IsAuthenticated]
    

class PatentSearchViewSet(HaystackViewSet):
    index_models = [Patent]
    serializer_class = PatentIndexSerializer
    permission_classes = [permissions.IsAuthenticated]


class AutocompleteSearchViewSet(HaystackViewSet):
    index_models = [Patent]
    serializer_class = AutocompleteSerializer
    filter_backends = [HaystackAutocompleteFilter]
    permission_classes = [permissions.IsAuthenticated]
