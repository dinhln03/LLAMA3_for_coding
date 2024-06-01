# coding=utf-8

from pub.tables.resources import *
from pub.tables.user import *
import pub.client.login as login
from pub.permission.user import is_logged,is_owner

def is_valid_key(key, r_type):

    try:
        resource_type.objects.get(key=key)
        return False
    except:
        pass

    try:
        resource_info.objects.get(key=key)
        return False
    except:
        pass

    if (r_type == -1):
        return True

    try:
        if(r_type==s.RESOURCE_TYPE_CUSTOMED):
            resource_customed.objects.get(key=key)
            return False
        elif(r_type == s.RESOURCE_TYPE_TEMPLATED):
            resource_templated.objects.get(key=key)
            return False
        elif(r_type == s.RESOURCE_TYPE_RESTFUL_API):
            resource_restful.objects.get(key=key)
            return False
        elif(r_type == s.RESOURCE_TYPE_IFRAME):
            resource_iframe.objects.get(key=key)
            return False
        elif(r_type == s.RESOURCE_TYPE_SHORT_LINK):
            resource_link.objects.get(key=key)
            return False
        else:
            return False
    except:
        return True

def set_permission(key,readable,writeable,modifiable,token=''):
    try:
        res = resource_permission.objects.get(key=key)
        res.delete()
        raise  Exception()
    except:
        resource_permission.objects.create(key=key,readable=readable,writeable=writeable,modifiable=modifiable,token=token)

def can_read(request,key,token=''):
    try:
        readable,_,_,verify_token =__get_resource_permission(key)
        return __accessibility_verfy(readable,request,key,token,verify_token)
    except:
        return False

def can_write(request,key,token=''):
    try:
        _,writeable,_,verify_token = __get_resource_permission(key)
        return __accessibility_verfy(writeable,request,key,token,verify_token)
    except:
        return False

def can_modify(request,key,token=''):
    try:
        _,_,modifiable,verify_token = __get_resource_permission(key)
        return __accessibility_verfy(modifiable,request,key,token,verify_token)
    except:
        return False

def can_create(request, r_type):

    if not is_logged(request):
        return False

    return True
    #
    # try:
    #     user = login.get_user_by_session(request,request.session.get(s.SESSION_LOGIN))
    # except:
    #     return False
    #
    # p = user_permission.objects.get(user_id=user, type=r_type).volume
    #
    # if p>0:
    #     return True
    #
    # return False

def did_create(request,r_type):

    if is_logged(request):

        user = login.get_user_by_session(request,request.session.get(s.SESSION_LOGIN))

        p = user_permission.objects.get(user_id=user, type=r_type)

        p.volume = p.volume - 1

        p.save()


def __get_resource_permission(key):
    p = resource_permission.objects.get(key=key)

    readable = p.readable

    writeable = p.writeable

    modifiable = p.modifiable

    token = p.token

    return readable, writeable, modifiable, token


def __accessibility_verfy(accessibility, request, key, token, verify_token):
    if accessibility == s.ACCESSIBILITY_PUBLIC:
        return True

    elif accessibility == s.ACCESSIBILITY_LOGIN or accessibility == s.ACCESSIBILITY_LOGIN_OR_TOKEN:
        if is_logged(request):
            return True
        else:
            if token != '':
                if token == verify_token:
                    return True

    elif accessibility == s.ACCESSIBILITY_PRIVATE:
        if is_logged(request):
            if is_owner(request, key):
                return True
        return False

    elif accessibility == s.ACCESSIBILITY_TOKEN:
        if token != '':
            if token == verify_token:
                return True
