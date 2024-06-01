from django.shortcuts import render,redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.urls.base import reverse
import datetime as dt
from .models import Profile,Project,Rating,User
from .forms import *
from .email import send_welcome_email


# Create your views here.



def signup_view(request):
    date = dt.date.today()
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            name=form.cleaned_data['fullname']
            email=form.cleaned_data['email']
            send_welcome_email(name,email,date)
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('main:home')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})





def home_page(request):
   
    projects = Project.objects.all()[1:]
    highlightProject = Project.objects.all().order_by('id').last()
    try:
        rating = Rating.objects.filter(project_id=highlightProject.id)
    except Rating.DoesNotExist:
        rating=None

    ctx={
        "projects":projects,
        "highlightProject":highlightProject,
        "rating":rating
        
    }
    return render(request,"main/home_page.html",ctx)

@login_required(login_url='/login')
def post_project(request):
    current_user = request.user
    user = Profile.objects.get(user=current_user)
    form = SubmitProjectForm()
    if request.method == 'POST':
        form = SubmitProjectForm(request.POST,request.FILES)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = user
            project.save()
            return redirect('/')
    else:
        form = SubmitProjectForm()
    ctx = {
        'form':form
    }
    
    return render(request,"main/post_project.html",ctx)

@login_required(login_url='/login')
def project_view(request,id):
    user = Profile.objects.get(user= request.user)
    project = Project.objects.get(id=id)
    
    ratings=Rating.objects.filter(project = project).last()
    tech_tags = project.technologies.split(",")
 
    try:
        rates = Rating.objects.filter(user=user,project=project).first()
    except Rating.DoesNotExist:
        rates=None
        
    if rates is None:
        rates_status=False
    else:
        rates_status = True
   
    form = RateForm()
    rating=None
    if request.method == 'POST':
        form = RateForm(request.POST)
        if form.is_valid():
            rate = form.save(commit=False)
            rate.user = user
            rate.project = project
            rate.save()
        try:
            rating = Rating.objects.filter(project_id=id)
        except Rating.DoesNotExist:
            rating=None
        design = form.cleaned_data['design']
        usability = form.cleaned_data['usability']
        content = form.cleaned_data['content']
        rate.average = (design + usability + content)/2
        rate.save()
        
        design_ratings = [d.design for d in rating]
        design_average = sum(design_ratings) / len(design_ratings)

        usability_ratings = [us.usability for us in rating]
        usability_average = sum(usability_ratings) / len(usability_ratings)

        content_ratings = [content.content for content in rating]
        content_average = sum(content_ratings) / len(content_ratings)
        score = (design_average + usability_average + content_average) / 3

        rate.design_average = round(design_average, 2)
        rate.usability_average = round(usability_average, 2)
        rate.content_average = round(content_average, 2)
        rate.score = round(score, 2)
    
        rate.save()
        return redirect("main:project_view", id=project.id)
    else:
        form = RateForm()
              
    ctx={
        "project":project,
        "ratings":ratings,
        "form":form,
        "tech_tags":tech_tags,
        "rates_status":rates_status 
    }
    return render(request,"main/view_project.html",ctx)

@login_required(login_url='/login')
def search_results(request):
    if 'search_project' in request.GET and request.GET["search_project"]:
        search_term = request.GET.get("search_project")
        searched_projects = Project.search_project_by_search_term(search_term)
       
        message = f"{search_term}"
        return render(request, 'main/search.html', {"message":message,"projects": searched_projects})
    else:
        message = "You haven't searched for any project"
    return render(request, 'main/search.html', {'message': message})
    
@login_required(login_url='/login')
def user_profile(request,username):
    current_user = request.user
    user_selected= User.objects.get(username=username)
    user_profile = Profile.filter_profile_by_id(user_selected.id)
    projects = Project.objects.filter(user=user_profile)
    if request.user == user_selected:
        return redirect('main:profile', username=username)
        
    ctx={
        "user_profile":user_profile,
        "projects":projects,
       
    }
    return render (request,'main/user_profile.html',ctx)

@login_required(login_url='/login') 
def profile(request,username):
    user= User.objects.get(username=username)
    user_profile = Profile.filter_profile_by_id(user.id) 
    projects = Project.objects.filter(user=user_profile)
    ctx={
        "user_profile":user_profile,
        "user":user,
         "projects":projects,
    }
    return render (request,'profile/profile.html',ctx)


@login_required
def update_profile(request,username):
    user= User.objects.get(username=username)
    profile = Profile.filter_profile_by_id(user.id) 
    form = UpdateUserProfileForm(instance=profile)
    if request.method == "POST":
            form = UpdateUserProfileForm(request.POST,request.FILES,instance=profile)
            if form.is_valid():  
                profile = form.save(commit=False)
                profile.save()
                return redirect('main:profile' ,username=username) 
            
    ctx= {"form":form}
    return render(request, 'profile/update_profile.html',ctx)


    
    
    
