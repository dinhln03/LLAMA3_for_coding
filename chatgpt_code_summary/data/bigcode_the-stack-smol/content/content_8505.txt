from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms.models import ModelForm
from .models import Image,Comment,Profile


class RegisterUserForm(UserCreationForm):
    email = forms.EmailField()
    first_name = forms.CharField(max_length=50)
    last_name = forms.CharField(max_length=50)
    
    
    class  Meta:
        model = User
        fields = ('username','first_name','last_name','email','password1','password2')
        
class ImageForm(ModelForm):
    
    
    class Meta:
        model = Image
        fields = ['image','name','caption','location' ]
        
class ProfileForm(ModelForm):
    
    
    class Meta:
        model = Profile
        fields = ['bio','profile_pic' ]
        
        
class CommentForm(ModelForm):
    
    class Meta:
        model = Comment
        fields =['content']
        widgets = {
            'content':forms.Textarea(attrs={'rows':2,'placeholder':'write comment'}),
        }