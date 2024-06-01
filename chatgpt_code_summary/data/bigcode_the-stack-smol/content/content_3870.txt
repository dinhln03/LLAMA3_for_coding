from django.shortcuts import render
from django.http import HttpResponse


def feedHome(request):
    return  HttpResponse('<p>Welcome To My Social App</p>')