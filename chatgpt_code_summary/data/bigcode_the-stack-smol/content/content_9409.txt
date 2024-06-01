from django.contrib.auth.decorators import permission_required
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from catalog import models as cmod
from django_mako_plus import view_function, jscontext
import requests
import json
# @permission_required('manager')
@view_function
def process_request (request):
    category_name = request.GET.get('category')
    product_name = request.GET.get('name')
    max_price = request.GET.get('max_price')
    page = request.GET.get('page')
    if page is not None:
        pnum = int(page)
    else:
        pnum = 1
    products = []
    qry = cmod.Product.objects.all()
    if product_name:
        qry = qry.filter(name__icontains=product_name)
    if max_price:
        qry = qry.filter(price__lte=max_price)
    if category_name:
        qry = qry.filter(category__name__icontains=category_name)
    qry = qry.order_by('category','name')
    for p in qry:
        item = {
                'category': p.category.name,
                'name': p.name,
                'price': p.price,
                }
        products.append(item)
    products = products[(pnum - 1)*6:pnum*6]
    return JsonResponse(products, safe=False)
