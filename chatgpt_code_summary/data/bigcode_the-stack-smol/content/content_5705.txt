# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render, get_object_or_404

from .forms import PostForm, CommentForm
from .models import Post, Comment


def post_list(request):
    queryset_list = Post.objects.all().order_by('-publish', 'id')

    paginator = Paginator(queryset_list, 25)  # Show 25 contacts per page

    page = request.GET.get('page')
    try:
        post_list = paginator.page(page)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        post_list = paginator.page(1)
    except EmptyPage:
        # If page is out of range (e.g. 9999), deliver last page of results.
        post_list = paginator.page(paginator.num_pages)

    return render(request, "pages/home.html", {
        'post_list': post_list,
    })


def post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug)

    if request.method == 'POST':
        if request.user:
            form = CommentForm(request.POST)
            if form.is_valid():
                instance = form.save(commit=False)
                instance.user = request.user
                instance.post = post
                instance.save()
                messages.add_message(request, messages.SUCCESS, 'Comment Added')
    form = CommentForm()

    return render(request, 'blog/post_detail.html', {
        'post': post,
        'form': form,
    })


@login_required
def post_add(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user
            instance.save()
            messages.add_message(request, messages.SUCCESS, 'Blog Post Added')
            form = PostForm()
    else:
        form = PostForm()
    return render(request, 'blog/post_form.html', {
        'form': form,
    })
