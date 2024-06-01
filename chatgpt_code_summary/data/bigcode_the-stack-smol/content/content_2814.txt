from django.views.generic import TemplateView, CreateView, UpdateView
from django.urls import reverse_lazy
from home_app import forms
from django.contrib.auth.mixins import LoginRequiredMixin
from account_app.models import CustomUser

# Create your views here.


class IndexView(TemplateView):
    template_name = 'home_app/index.html'


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = 'home_app/profile.html'


class RegistrationView(CreateView):
    form_class = forms.UserCreateForm
    success_url = reverse_lazy('home-app:index')
    template_name = 'registration/registration.html'


class UserUpdateView(UpdateView):
    form_class = forms.UserUpdateForm
    success_url = reverse_lazy('home-app:profile')
    template_name = 'registration/registration_form.html'
    model = CustomUser


class Page403View(TemplateView):
    template_name = 'home_app/403.html'
