#!/usr/bin/env python
from django.urls import reverse_lazy
from django.shortcuts import Http404
from django.utils.translation import ugettext as _
from vanilla import ListView, CreateView, DetailView, UpdateView, DeleteView, TemplateView
from .forms import ArticleForm, ArticleSearchForm
from .models import Article, Folder
from haystack.generic_views import SearchView
from haystack.query import SearchQuerySet


class ArticleList(ListView):
    model = Article
    paginate_by = 20


#class ArticleCreate(CreateView):
#    model = Article
#    form_class = ArticleForm
#    success_url = reverse_lazy('bibloi:list')


class ArticleDetail(DetailView):
    model = Article

    def get_context_data(self, **kwargs):
        context = super(ArticleDetail, self).get_context_data(**kwargs)
        return context

#class ArticleUpdate(UpdateView):
#    model = Article
#    form_class = ArticleForm
#    success_url = reverse_lazy('bibloi:list')


#class ArticleDelete(DeleteView):
#    model = Article
#    success_url = reverse_lazy('bibloi:list')

class ArticleSearch(SearchView):
    template_name = 'search/search.html'
    form_class = ArticleSearchForm
    queryset = SearchQuerySet().order_by('-date')
    paginate_by = 5

    def get_context_data(self, **kwargs):
        context = super(ArticleSearch, self).get_context_data(**kwargs)
        return context


class FolderView(ListView):
    model = Article
    template_name = 'bibloi/folder_browse.html'
    parent = None

    def get_queryset(self):
        path = self.kwargs.get('path', '')
        folders = path.split('/')

        for folder in folders:
            try:
                if not self.parent:
                    if folder:
                        self.parent = Folder.objects.get(name=folder)
                else:
                    self.parent = self.parent.get_children().get(name=folder)
            except Folder.DoesNotExist:
                raise Http404(_('Folder does not exist'))

        return self.model.objects.filter(folder=self.parent)

    def get_context_data(self, **kwargs):
        context = super(FolderView, self).get_context_data(**kwargs)

        context['parent_folders'] = self.parent.parent_folders if self.parent else []
        context['current_folder'] = self.parent
        if self.parent:
            context['folders'] = self.parent.get_children()
        else:
            context['folders'] = Folder.objects.filter(parent=self.parent)
        return context


class TasksView(TemplateView):
    template_name = 'tasks.html'
