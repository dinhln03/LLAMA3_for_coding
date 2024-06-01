from django import forms
from django.contrib.auth.models import User
from django.forms import ModelForm
from artapp.models import Artist, Art
from django.template.defaultfilters import slugify


class RegistrationForm(ModelForm):
	username 	= forms.CharField(label=(u'User Name'))
	email 		= forms.EmailField(label=(u'Email Address'))
	password 	= forms.CharField(label=(u'Password'), widget=forms.PasswordInput(render_value=False))
	password1 	= forms.CharField(label=(u'Verify Password'), widget=forms.PasswordInput(render_value=False))
	
	class Meta:
		model = Artist
		exclude = ('user',)

	def clean_username(self):
		username = self.cleaned_data['username']
		try:
			User.objects.get(username=username)
		except User.DoesNotExist:
			return username
		raise forms.ValidationError('That username is already taken, please select another')	
	
	def clean(self):
		if self.cleaned_data['password'] != self.cleaned_data['password1']:
			raise forms.ValidationError("The passwords did not match. Please try again")
		myslug=slugify(self.cleaned_data['username'])
		try:
			Artist.objects.get(slug=myslug)
		except Artist.DoesNotExist:
			return self.cleaned_data
			
			#return self.cleaned_data
		raise forms.ValidationError("That username is already taken, please select another")
		 
class LoginForm(forms.Form):
	username 	= forms.CharField(label=(u'User Name'))
	password 	= forms.CharField(label=(u'Password'), widget=forms.PasswordInput(render_value=False))
	#next_url	= forms.CharField(label=(u'next url'), widget=forms.HiddenInput()) #hidden

class VoteForm(forms.Form):
	VOTECHOICES = [('upvote', 'downvote')]
	vote = forms.MultipleChoiceField(required=False, widget=forms.CheckboxSelectMultiple, choices=VOTECHOICES)
	artslug = forms.CharField()
	
class SubmitArtForm(ModelForm):
	
	class Meta:
		model = Art
		fields = ['title', 'drawing']
		#exclude = ('slug','created_at', 'likes',)
	
	def clean(self):
		myslug=slugify(self.cleaned_data['title'])
		try:
			Art.objects.get(slug=myslug)
		except Art.DoesNotExist:
			return self.cleaned_data
		raise forms.ValidationError("Ascii Art with that slug already exists, please pick another title")
"""
	def clean_title(self):
		title = self.cleaned_data['title']
		try:
			Art.objects.get(title=title)
		except Art.DoesNotExist:
			return title
		raise forms.ValidationError("Ascii Art with that title already exists, please pick another title")
"""