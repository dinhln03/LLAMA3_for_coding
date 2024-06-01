from django.shortcuts import render,redirect
from .models import Image,Location,Category
# Create your views here.
def intro(request):
    images = Image.objects.all()

    return render(request, 'intro.html',{'images':images})
def search_results(request):
    if 'image' in request.GET and request.GET["image"]:
        search_term = request.GET.get("image")
        searched_images = Image.search_by_cate(search_term)
        message = f"{search_term}"
        return render(request,'search.html',{"message":message,"images":searched_images})
