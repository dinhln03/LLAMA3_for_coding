from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(request):
    category_list = Category.objects.order_by('-name')[:5]

    #category_list = Category.objects().order_by('-name')
    context = {'categories': category_list}   # Aquí van la las variables para la plantilla

    return render(request,'index.html', context)
