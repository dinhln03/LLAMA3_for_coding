from rest_framework import generics
from ..models import Article, Country, Source
from .serializers import ArticleSerializer, CountrySerializer, SourceSerializer


class ArticleListView(generics.ListCreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = []


class ArticleDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = []


class CountryListView(generics.ListCreateAPIView):
    queryset = Country.objects.all()
    serializer_class = CountrySerializer
    permission_classes = []


class CountryDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Country.objects.all()
    serializer_class = CountrySerializer
    permission_classes = []


class SourceListView(generics.ListCreateAPIView):
    queryset = Source.objects.all()
    serializer_class = SourceSerializer
    permission_classes = []


class SourceDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Source.objects.all()
    serializer_class = SourceSerializer
    permission_classes = []
