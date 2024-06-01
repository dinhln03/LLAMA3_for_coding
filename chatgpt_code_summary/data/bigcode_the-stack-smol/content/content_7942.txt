from django.shortcuts import render


def home(request):
    """
    View function for simply rendering the Ionic Angular
    index.html
    """
    return render(request, 'www/index.html')
