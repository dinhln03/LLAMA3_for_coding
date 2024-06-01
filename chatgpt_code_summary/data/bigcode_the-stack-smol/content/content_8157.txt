from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.views.generic import View, TemplateView, FormView

def Index(request):
    if request.method == 'POST':
        if 'indexLogin' in request.POST:
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('goal_app:goals')
            else:
                login_form = AuthenticationForm()
                signup_form = UserCreationForm()
                errorMessage = True
                context = {"intro_page": "active", 'login_form': login_form, 'errorMessage': errorMessage, 'signup_form': signup_form,}
        elif 'indexSignup' in request.POST:
            signup_form = UserCreationForm(request.POST)
            if signup_form.is_valid():
                signup_form.save()
                username = signup_form.cleaned_data.get('username')
                raw_password = signup_form.cleaned_data.get('password1')
                user = authenticate(username=username, password=raw_password)
                login(request, user)
                return redirect('goal_app:goals')
            else:
                login_form = AuthenticationForm()
                context = {"intro_page": "active", 'login_form': login_form, 'signup_form': signup_form, }
    else:
        login_form = AuthenticationForm()
        signup_form = UserCreationForm()
        context = {"intro_page": "active", 'login_form': login_form, 'signup_form': signup_form, }
    return render(request, 'index.html', context)

def About(request):
    context = {"about_page": "active"}
    return render(request, 'about.html', context)

def Signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('goal_app:goals')
    else:
        context = {'signup_page': 'active'}
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})
