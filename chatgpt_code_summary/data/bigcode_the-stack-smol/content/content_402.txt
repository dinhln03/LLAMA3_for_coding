from django.shortcuts import render
from wiki.models import Page
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.shortcuts import get_object_or_404,render


class PageList(ListView):
    """
    This view grabs all the pages out of the database
    returns a list of each unique wiki page for the
    user to access on the website through 'list.html'
    """
    model = Page

    def get(self, request):
        """ Returns a list of wiki pages. """
        pages = Page.objects.all()
        context = {'pages': pages}
        return render(request, 'list.html', context=context)

class PageDetailView(DetailView):
    """
    This view returns a page for a unique wiki using it's slug as an identifier
    or a 404 message if the page does not exist
    """
    model = Page
    
    def get(self, request, slug):
      wiki = get_object_or_404(Page, slug=slug)
      return render(request, 'page.html', {'wiki': wiki})

    def post(self, request, slug):
        pass
