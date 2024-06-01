from django import forms

class SubmitEmbed(forms.Form):
    url = forms.URLField()