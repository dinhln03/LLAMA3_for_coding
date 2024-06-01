# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# IntegrityError Exception for checking duplicate entry, 
# connection import to establish connection to database 
from django.db import IntegrityError, connection 

# Used for serializing object data to json string
from django.core.serializers.json import DjangoJSONEncoder 
from django.core.serializers import serialize

# Django HTTP Request
from django.shortcuts import render, redirect
from django.http import HttpResponse, Http404, HttpResponseForbidden, HttpResponseRedirect, JsonResponse

# Generic views as Class
from django.views.generic import TemplateView
from django.views.generic.list import ListView
from django.views import View

# system imports
import sys, os, csv, json, datetime, calendar, re

# Django utils
from django.utils import timezone, safestring
from django.utils.decorators import method_decorator

# Django authentication
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password

# Django Messaging Framework
from django.contrib import messages

# Conditional operators and exception for models
from django.db.models import Q, Count, Sum, Prefetch
from django.core.exceptions import ObjectDoesNotExist

# Paginator class import
from django.core.paginator import Paginator, InvalidPage, EmptyPage, PageNotAnInteger

# Helpers
import app.user_helper as user_helper
import app.records_helper as records_helper

# Forms
from app.forms import *


#=========================================================================================
#   GET SUB CATEGORY ON BASIS OF CATEGORY
#=========================================================================================

def get_sub_category(request):
    sub_cat_list = request.GET.getlist("cat_id[]")
    
    if len(sub_cat_list) > 0:
        sub_cats = records_helper.SubCategoryList(sub_cat_list)
        
        html = []
        for sub in sub_cats:
            html.append('<option value="'+str(sub.id)+'">'+str(sub)+'</option>');
        return HttpResponse(''.join(html))
    return HttpResponse('')   