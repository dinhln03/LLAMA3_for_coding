from django.shortcuts import render
from django.views.generic.detail import DetailView

from todos.models import Task


def index(request):
    return render(request, 'frontend/index.html')


class TodoDetailView(DetailView):
    model = Task
    template_name = 'frontend/index.html'
