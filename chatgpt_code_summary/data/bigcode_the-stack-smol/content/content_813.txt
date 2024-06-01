from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from Constants import *
from Gifts.getRecommendations.RS import Users, Recommendations
import json

# Create your views here.


def check_input(request, mandatory_fields, optional_fields=None):
    if not optional_fields: optional_fields = []
    for key in request.keys():
        if key not in mandatory_fields and key not in optional_fields:
            return {'result': 'Error', 'message': key + ' is not a valid field'}
    for field in mandatory_fields:
        if field not in request.keys():
            return {'result': 'Error', 'message': field + ' do not presented'}
    return {"result": "Success"}


def add_user(request):
    if 'userProfile' not in request:
        return JsonResponse({'result': 'Error', 'message': 'userProfile do not presented'})

    result = check_input(request['userProfile'], ["sex", "age", "hobbies", "userType"],
                         ["alreadyGifted", "lovedCategories"])
    if result['result'] == "Error":
        return JsonResponse(result)

    if request['userProfile']['sex'] not in ['Female', 'Male']:
        return JsonResponse({'result': 'Error', 'message': request['userProfile']['sex'] +
                                                           ' is not a valid sex'})

    if 'alreadyGifted' not in request['userProfile']:
        request['userProfile']['alreadyGifted'] = []
    if 'lovedCategories' not in request['userProfile']:
        request['userProfile']['lovedCategories'] = []

    try:
        user_id = Users.add_user(request['userProfile'])

    except Exception as e:
        print e
        return JsonResponse({'result': 'Error', 'message': 'error while adding user'})

    return JsonResponse({'result': 'Success', 'data': {'userId': user_id}})


def make_list(request):
    result = check_input(request, ["userId"], ["filter"])
    if result['result'] == "Error":
        return JsonResponse(result)
    if 'filter' in request:
        result = check_input(request['filter'], [], ["minPrice", "maxPrice"])
        if result['result'] == "Error":
            return JsonResponse(result)

    min_price = None
    max_price = None
    if 'filter' in request:
        if 'minPrice' in request['filter']:
            min_price = request['filter']['minPrice']
        if 'maxPrice' in request['filter']:
            max_price = request['filter']['maxPrice']
    try:
        Recommendations.generate_list(request['userId'], min_price, max_price)
        number_of_pages = Recommendations.get_number_of_pages(request['userId'])
    except Exception as e:
        print e
        return JsonResponse({'result': 'error', 'message': 'error while making list'})
    return JsonResponse({'result': 'Success', 'data': {'numberOfPages': number_of_pages}})


def get_suggestions(request):
    result = check_input(request, ["page", "userId"])
    if result['result'] == "Error":
        return JsonResponse(result)
    try:
        items = Recommendations.get_page(request['userId'], request['page'])
        number_of_pages = Recommendations.get_number_of_pages(request['userId'])
    except Exception as e:
        print e
        return JsonResponse({'result': 'Error', 'message': 'error during getting list'})
    if items:
        request = {'result': 'Success', 'data': {'items': items, "numberOfPages": number_of_pages}}
    elif items == []:
        request = {'result': 'Error', 'message': 'page out of range'}
    else:
        request = {'result': 'Error', 'message': 'error during getting list'}
    return JsonResponse(request)


def rate_item(request):
    result = check_input(request, ["userId", "itemId", "rating"])
    if result['result'] == "Error":
        return JsonResponse(result)
    try:
        Recommendations.rate_and_remove(request['userId'], request['itemId'], request['rating'])
        number_of_pages = Recommendations.get_number_of_pages(request['userId'])

    except Exception as e:
        print e
        return JsonResponse({"result": "Error", "message": "error during rating item"})
    return JsonResponse({"result": "Success", 'data': {'numberOfPages': number_of_pages}})


@csrf_exempt
def home(request):
    if request.method == "POST":
        try:
            request_dict = json.loads(request.body)
            print(request_dict)
            if 'task' not in request_dict:
                return JsonResponse({'result': 'Error', 'message': 'task do not presented'})
            if 'data' not in request_dict:
                return JsonResponse({'result': 'Error', 'message': 'data do not presented'})
            if request_dict['task'] == 'addUser':
                return add_user(request_dict['data'])
            if request_dict['task'] == 'makeList':
                return make_list(request_dict['data'])
            if request_dict['task'] == 'getSuggestions':
                return get_suggestions(request_dict['data'])
            if request_dict['task'] == 'rateItem':
                return rate_item(request_dict['data'])
            return JsonResponse({'result': 'Error', 'message':
                request_dict['task'] + " is not a valid task"})
        except Exception as e:
            print e
            return JsonResponse({'result': 'Error', 'message': "strange error"})

    return HttpResponse('''
        <h1>Welcome on GRS</h1>
    ''')



