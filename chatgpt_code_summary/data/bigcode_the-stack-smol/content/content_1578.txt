from django.shortcuts import render
from accounts import models
from rest_framework import viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated 
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.settings import api_settings

from accounts import serializers # Will use this to tell API what data to exect when making a POST PUT PATCH request to API
from accounts import models
from accounts import permissions

class Injection_DetailsViewSet(viewsets.ModelViewSet):
    """Handles creating, reading and updating patient info readings"""
    
    authentication_classes = (TokenAuthentication,)
    serializer_class = serializers.Injection_DetailsSerializer # This points to the
    queryset = models.Injection_Details.objects.all()
    permission_classes = (permissions.UpdateOwnReading, IsAuthenticated,) # Validates that a user is authenticated to read or modify objects

    def get_queryset(self):
        user = self.request.user
        return models.Injection_Details.objects.get_queryset().filter(user_profile=user)
    
    

    def perform_create(self, serializer): # overriding this function so that when a user tries to create an object they are validated as the current user
        """Sets the patient profile to the logged in user"""
        serializer.save(user_profile=self.request.user) # This sets the user profile to the current user from the serializer passed in 
    
    #def create(self, serializer): # overriding this function so that when a user
        #patient_info = models.PatientInfo.objects.filter(user_profile=self.request.user)
        #serializer.save = self.get_serializer(patient_info, many = True) # This sets the user profile to the current user from the serializer passed in 
        #serializer.is_valid(raise_exceptions=True)
        #self.perform_create(serializer)
        #return Response(serializer.data)