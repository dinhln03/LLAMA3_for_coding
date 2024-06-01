from django.shortcuts import render, get_object_or_404
from .models import PostReview
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


def single_review(request, year, month, day, review):
    review = get_object_or_404(PostReview, slug=review,
                               status='published',
                               publish__year=year,
                               publish__month=month,
                               publish__day=day)
    return render(request, 'reviews/review/single.html', {'review': review})


def review_list(request):
    list_object = PostReview.published.all()
    paginator = Paginator(list_object, 1)
    page = request.GET.get('page')
    try:
        reviews = paginator.page(page)
    except PageNotAnInteger:
        reviews = paginator.page(1)
    except EmptyPage:
        reviews = paginator.page(paginator.num_pages)
    return render(request,
                  'reviews/review/list.html',
                  {'page': page,
                   'reviews': reviews})
