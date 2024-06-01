# -*- coding: utf-8 -*-
from django.db import models
from django.shortcuts import render,redirect
from django.views import View
from django.contrib.auth import authenticate, login , logout as django_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import Http404, HttpResponseRedirect
from django.core.paginator import Paginator


import django_filters
from django import forms

from .models import B2BRecord

# Create your views here.


class LendRecordFilter(django_filters.FilterSet):
	created = django_filters.DateFromToRangeFilter(
		widget=forms.SplitDateTimeWidget(
			attrs={
				'class':'datepicker',
				'type':'date',
			}
		)
	)

	class Meta:
		model = B2BRecord
		fields = ['depot','created']

class LendRecordListView(View):
	def get(self,request):
		ft = LendRecordFilter(request.GET)
		merchandises = ft.qs
		paginator = Paginator(merchandises, 15)
		page_number = request.GET.get('page')
		page_obj = paginator.get_page(page_number)

		return render(request,'lend_record_list.html',{'records':page_obj,'filter':ft})