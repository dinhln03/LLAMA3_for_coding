# -*- coding: utf-8 -*-
# 留言板

# 1、新建目录下一定要有__init__.py文件，否则不能被其它文件引用、不能沿路径读写文件。from ... 。
# 2、urls.py中,设置第一级路由名ask。 在.../mysite/mysite/urls.py中  url(r'^ask/', include('account.ask.urls')),
# 3、admin.py中,设置数据库显示。在.../mysite/account/admin.py中 @admin.register(Technologyask) ...
# 4、templates中,增加模板文件目录/ask
import datetime
import os
import json
from django.shortcuts import render
from django.http.response import HttpResponseRedirect,HttpResponse
from . models import Guestbook,Reply
from django.contrib.auth.decorators import login_required  #使用注意在settings.py中设置 LOGIN_URL = '/login/'
from django.contrib.auth.models import User
from myAPI.pageAPI import djangoPage
from django.contrib import messages

PAGE_NUM = 20 #每页显示数
# http://localhost:9000/guestbook/reply/
#@login_required
def reply(request):
    if request.method != 'POST': 
        return render(request, 'guestbook/reply.html', context=locals())
    title = request.POST['title']
    content = request.POST['content']
    Guestbook.objects.filter(title=title).update(state=1)#更改回答状态    
    if request.user.username == 'admin': #admin回复
        Reply.objects.filter(title=title).update(content=content )
        Reply.objects.filter(title=title).update(username = 'admin' )
        Reply.objects.filter(title=title).update(date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") )   
    return HttpResponseRedirect('/guestbook/showreply/')
  
@login_required
def gettitle(request):
    title = request.GET.get('title','')
    if title == '':
        return HttpResponse('no')  
    return render(request, 'guestbook/reply.html', context=locals())
    
# http://localhost:9000/guestbook/create/    
@login_required
def create(request):
    if request.method != 'POST': 
        return render(request, 'guestbook/create.html', context=locals())
    title = request.POST['title']
    content = request.POST['content']
    istitle = Guestbook.objects.filter(title = title)
    if istitle:  
        messages.info(request, '告警：标题 '+ title + '已经被使用！')
        return HttpResponseRedirect('/guestbook/show/')   

    if content:
        guestbooks = Guestbook(username=request.user,title=title,content=content)   
        guestbooks.save()
        guestbookname = Guestbook.objects.get(title=title).username
        replys = Reply(guestbookname=guestbookname,title=title)   
        replys.save()
    else:
        messages.info(request,'告警：留言内容为空！')   
    return HttpResponseRedirect('/guestbook/show/')

# http://localhost:9000/guestbook/show/    
@login_required
def show(request, page):
    if request.user.is_superuser:
        guestbooks = Guestbook.objects.filter().order_by('-date','-id') 
        guestbooks, pageList, paginator, page = djangoPage(guestbooks,page,PAGE_NUM)  #调用分页函数
        replys = Reply.objects.filter(guestbookname=request.user.username).order_by('-date', '-id') 
        offset = PAGE_NUM * (page - 1)
        return render(request, 'guestbook/showall.html', context=locals())    
   
    guestbooks = Guestbook.objects.filter(username=request.user.username).order_by('-date', '-id') 
    guestbooks, pageList, paginator, page = djangoPage(guestbooks,page,PAGE_NUM)  #调用分页函数
    replys = Reply.objects.filter(guestbookname=request.user.username).order_by('-date', '-id') 
    offset = PAGE_NUM * (page - 1)
    return render(request, 'guestbook/show.html', context=locals())
# http://localhost:9000/guestbook/showreply/  
@login_required
def showreply(request, page):
    title = request.GET.get('title','')
    if title != '':
        replys = Reply.objects.filter(title=title)
    else:
        replys = Reply.objects.filter(username=request.user).order_by('-date', '-id') 
    replys, pageList, paginator, page = djangoPage(replys,page,PAGE_NUM)  #调用分页函数
       
    offset = PAGE_NUM * (page - 1)       
    return render(request, 'guestbook/showreply.html', context=locals())
    