from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template

from mainsite.forms import NewsCreate
from mainsite.models import News, ContactForm, Issue


@login_required(login_url="/admin-panel/login/")
def index(request):
    context = {}
    context['segment'] = 'index'

    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/admin-panel/login/")
def profile(request):
    context = {}
    context['segment'] = 'profile'

    html_template = loader.get_template('page-user.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/admin-panel/login/")
def news(request):
    list = News.objects.all()
    context = {"list": list}
    context['segment'] = 'news'

    html_template = loader.get_template('news.html')
    return HttpResponse(html_template.render(context, request))


def add_news(request):
    upload = NewsCreate()
    if request.method == 'POST':
        upload = NewsCreate(request.POST, request.FILES)
        if upload.is_valid():
            upload.save()
            return redirect('/admin-panel/news')
        else:
            return HttpResponse(
                """your form is wrong, reload on <a href = "{{ url : '/admin-panel/news'}}">reload</a>""")
    else:
        context = {
            "upload_form": upload,
            "action": "Добавить"
        }
        return render(request, 'add-news.html', context)


@login_required(login_url="/admin-panel/login/")
def update_news(request, news_id: int):
    try:
        news_sel = News.objects.get(pk=news_id)
    except news.DoesNotExist:
        return redirect('/admin-panel/news')
    news_form = NewsCreate(request.POST, request.FILES or None, instance=news_sel)
    if news_form.is_valid():
        news_form.save()
        return redirect('/admin-panel/news')
    context = {
        "ProductForm": news_form,
        "ProductModel": news_sel,
        "action": "Обновить"
    }
    return render(request, 'add-news.html', context)


@login_required(login_url="/admin-panel/login/")
def delete_news(request, news_id):
    news_id = int(news_id)
    try:
        news_sel = News.objects.get(pk=news_id)
    except news_id.DoesNotExist:
        return redirect('/admin-panel/news')
    news_sel.delete()
    return redirect('/admin-panel/news')


@login_required(login_url="/admin-panel/login/")
def contactforms(request):
    list = ContactForm.objects.all()
    context = {"list": list}
    context['segment'] = 'contactforms'

    html_template = loader.get_template('contact-forms.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/admin-panel/login/")
def requests(request):
    list = Issue.objects.all()
    context = {"list": list}
    context['segment'] = 'requests'

    html_template = loader.get_template('requests.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/admin-panel/login/")
def delete_contact_form(request, contact_id):
    contact_id = int(contact_id)
    try:
        contact_sel = ContactForm.objects.get(pk=contact_id)
    except contact_id.DoesNotExist:
        return redirect('/admin-panel/contacts')
    contact_sel.delete()
    return redirect('/admin-panel/contacts')


@login_required(login_url="/admin-panel/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]
        context['segment'] = load_template

        html_template = loader.get_template(load_template)
        return HttpResponse(html_template.render(context, request))
    except template.TemplateDoesNotExist:
        html_template = loader.get_template('page-404.html')
        return HttpResponse(html_template.render(context, request))
    except:
        html_template = loader.get_template('page-500.html')
        return HttpResponse(html_template.render(context, request))
