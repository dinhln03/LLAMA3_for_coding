# -*- coding: utf-8 -*-
from django import forms
from django.contrib.auth import get_user_model
from django.utils.translation import ugettext_lazy as _

from tcms.core.utils import string_to_list
from tcms.core.forms.fields import UserField
from tcms.management.models import Product, Version, Build
from tcms.testplans.models import TestPlan
from tcms.testcases.models import TestCase


# =========== Forms for search/filter ==============

class SearchProductForm(forms.Form):
    """
        Includes *only* fields used in search.html b/c
        the actual search is now done via JSON RPC.
    """
    name_product = forms.CharField(label='Product', max_length=100, required=False)