from django.http.response import HttpResponseRedirect
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from itertools import chain
from .models import Image, Profile, Comment
from .forms import NewProfileForm, NewImageForm
import string
import random

# Create your views here.


def generateNewName():

    letters = string.ascii_letters
    numbers = string.digits
    specialCharacters = string.punctuation


    acceptablePasswordCharacters = letters + numbers + specialCharacters
    generatedPassword = "".join(random.choice(acceptablePasswordCharacters) for i in range(8))

    # print("Your generared password is: " +generatedPassword)

    return generatedPassword

@login_required(login_url='/accounts/login/')
def index(request):
    title = 'Moments: Feed'
    
    current_user = request.user
    
    current_profile = Profile.objects.filter(user_name = current_user.username).first()
    
        
    if current_profile:
        all_posts = list()
        users_posts = Image.objects.filter(profile_id = current_profile).all()
              
        user_posts_count = users_posts.count()
       
        folowing_count = len(current_profile.following)
       
        if folowing_count >= 0: 
            all_following_post = list()
            for item in current_profile.following:
                following_posts = Image.objects.filter(profile_id = item).all()
                all_following_post.append(following_posts)
           
            all_posts = list(chain(users_posts, all_posts))
        return render(request, 'dashboard.html', {'title': title, 'all_posts': all_posts, "profile": current_profile, 'post_count': user_posts_count})
    
    else:
        return redirect('Create Profile')
      

@login_required(login_url='/accounts/login/')
def upload(request):
    title = 'Upload New Post'
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
    form = NewImageForm(request.POST, request.FILES)
    
    if request.method == 'POST' and request.FILES['new_image']:
        
        if form.is_valid():
            new_post = request.FILES['new_image']
            new_caption = form.cleaned_data['image_caption']        

            new_upload = Image(image = new_post, image_name = generateNewName(), image_caption = new_caption, profile_id = current_profile, likes = 0,)
            new_upload.save()
            return redirect('Dashboard')
        
        else:
            form = NewImageForm()
        
    return render(request, 'upload.html', {'title': title, 'form': form,  "profile": current_profile })

@login_required(login_url='/accounts/login/')
def create_profile(request):
    title = 'Moments: Create New Profile'
    form = NewProfileForm(request.POST, request.FILES)
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
    
    if request.method == 'POST' and request.FILES['profile_photo']:
        
        if form.is_valid():
            new_profile_photo = request.FILES['profile_photo']
            new_bio = form.cleaned_data['bio']
               
            print(new_bio)            

            username = request.user.username
            date_joined = request.user.date_joined
            
            new_profile = Profile(profile_photo= new_profile_photo, bio=new_bio, user_name= username, following= [], followers = [], joined= date_joined)
            new_profile.save()
            
            return redirect('Dashboard')
        
        else:
            form = NewProfileForm()
    
    return render(request, 'create_profile.html', {'title': title, 'form': form,  "profile": current_profile})



@login_required(login_url='/accounts/login/')
def profile(request):
    title = 'Upload New Post'
    
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
    
    user_posts = Image.objects.filter(profile_id = current_profile).all
    
    return render(request, 'profile.html', {'title': title, 'profile': current_profile, 'posts': user_posts})


@login_required(login_url='/accounts/login/')
def view_profile(request, user_id):
    title = 'Upload New Post'
    
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
            
    found_user = Profile.objects.filter(id = user_id).first()
    user_posts = Image.objects.filter(profile_id = found_user).all()
    
    return render(request, 'view_profile.html', {'title': title, 'profile': current_profile, 'user_profile': found_user, 'posts': user_posts})


@login_required(login_url='/accounts/login/')
def search(request):
    title = 'Upload New Post'
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
        
    if 'search' in request.GET and request.GET['search']:
        search_query = request.GET.get('search')
        print(search_query)
        found_users = Profile.objects.filter(user_name = search_query).all()       
        
        return render(request, 'search.html', {'title': title, 'profile': current_profile, 'search_query': search_query, 'results': found_users})

    return render(request, 'search.html', {'title': title, 'profile': current_profile})


@login_required(login_url='/accounts/login/')
def view(request, post_id):
    title = 'Upload New Post'
    
    current_profile = Profile.objects.filter(user_name = request.user.username).first()
    
    found_image = Image.objects.filter(id = post_id).first()
    
    image_comments = Comment.objects.filter(image_id = found_image.id ).all()
    
    return render(request, 'view.html', {'title': title, 'post': found_image, "profile": current_profile,  'comments': image_comments})