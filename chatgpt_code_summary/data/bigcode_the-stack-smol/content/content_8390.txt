from django import forms
from django.forms import ModelForm
from .models import FoodHomePageModel,FullMenuPageModel


class FoodHomeForm(forms.ModelForm):
    
    class Meta:
        model = FoodHomePageModel
        exclude = ("company",'status',)
        
        
class FoodMenuForm(forms.ModelForm):
    
    class Meta:
        model = FullMenuPageModel
        exclude = ("company",'status',)