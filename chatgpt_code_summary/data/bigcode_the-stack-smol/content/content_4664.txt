from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

import requests

# Create your views here.

def cnn(request):
    return render(request, 'CNN/cnn.html')

def change(request):

    ########################################################################
    if request.method == 'POST' and request.FILES['origin']:
        myfile = request.FILES['origin']
        fs = FileSystemStorage('./bssets/inputs/') #defaults to   MEDIA_ROOT
        filename = fs.save(myfile.name, myfile)

        ###############################################################
        # # Here we know the file is in
        api_host = 'http://35.221.233.111:8000/'
        headers = {'Content-Type': 'application/json'}
        photo = './bssets/inputs/'+ filename # file name = file_name + randomNumber

        files = {'file': (filename, open(photo, 'rb'), 'image/jpeg')}
        response = requests.post(api_host, files=files)

        return JsonResponse(response, safe=False)


