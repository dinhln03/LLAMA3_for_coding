from alipay import AliPay
from django.core.paginator import Paginator
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.http import urlquote

from Qshop.settings import alipay_private_key_string, alipay_public_key_string
from Seller.views import getPassword


# Create your views here.

def loginValid(func):
    """
    :desc 闭包函数校验是否登录
    :param func:
    :return:
    """

    def inner(request, *args, **kwargs):
        email = request.COOKIES.get("user")
        s_email = request.session.get("user")
        if email and s_email and email == s_email:
            user = LoginUser.objects.filter(email=email).first()
            if user:
                return func(request, *args, **kwargs)
        return HttpResponseRedirect("/Buyer/login/")

    return inner


def login(request):

    if request.method == "POST":
        erremail = ""
        email = request.POST.get("email")
        pwd = request.POST.get("pwd")
        user = LoginUser.objects.filter(email=email).first()
        if user:
            db_password = user.password
            pwd = getPassword(pwd)
            if db_password == pwd:

                response = HttpResponseRedirect("/Buyer/index/", locals())
                response.set_cookie("user", user.email)
                response.set_cookie("user_id", user.id)
                response.set_cookie("username", urlquote(user.username))
                request.session["user"] = user.email
                return response
            else:
                errpwd = "密码不匹配"
        else:
            erremail = "该邮箱未注册"
    return render(request, "buyer/login.html", locals())


def register(request):
    errmsg = ""
    if request.method == "POST":
        username = request.POST.get("user_name")
        pwd = request.POST.get("pwd")
        email = request.POST.get("email")
        db_email = LoginUser.objects.filter(email=email).first()
        db_username = LoginUser.objects.filter(username=username).first()
        if not db_email:
            if not db_username:
                user = LoginUser()
                user.username = username
                user.password = getPassword(pwd)
                user.email = email
                user.save()
                return HttpResponseRedirect("/Buyer/login/", locals())
            else:
                errmsg = "用户名已存在"
        else:
            errmsg = "邮箱已注册"
    return render(request, "buyer/register.html", {"errmsg": errmsg})


def index(request):
    types = GoodsType.objects.all()
    goods_result = []
    for type in types:
        goods = type.goods_set.order_by("goods_pro_date")[0:4]
        if len(goods) >= 4:
            goods_result.append({type: goods})
    return render(request, "buyer/index.html", locals())


def logout(request):
    url = request.META.get("HTTP_REFERER", "/Buyer/index/")
    response = HttpResponseRedirect(url)
    cookies = request.COOKIES.keys()
    for cookie in cookies:
        response.delete_cookie(cookie)
    if request.session.get("user"):
        del request.session['user']
    return response


@loginValid
def user_info(request):
    id = request.COOKIES.get("user_id")
    if id:
        user = LoginUser.objects.filter(id=id).first()
    return render(request, "buyer/user_center_info.html", locals())


@loginValid
def user_site(request):
    id = request.COOKIES.get("user_id")
    if id:
        user = LoginUser.objects.filter(id=id).first()
    return render(request, "buyer/user_center_site.html", locals())


@loginValid
def user_order(request):
    id = request.COOKIES.get("user_id")
    if id:
        user = LoginUser.objects.filter(id=id).first()
        order_lists = PayOrder.objects.filter(order_user=user).order_by("-order_date")

    return render(request, "buyer/user_center_order.html", locals())


"""
def good_list(request):
    type = request.GET.get("type")
    keyword = request.GET.get("keyword")
    goods_list = []
    if type == 'byid':
        if keyword:
            types = GoodsType.objects.get(id=keyword)
            goods_list = types.goods_set.order_by("goods_pro_date")
    elif type == 'bykey':
        if keyword:
            goods_list = Goods.objects.filter(goods_name__contains=keyword).order_by("goods_pro_date")
    if goods_list:
        nums = goods_list.count()
        nums = int(math.ceil(nums / 5))
        recommon_list = goods_list[:nums]
    return render(request, "buyer/goods_list.html", locals())

"""


def good_list(request, page):
    page = int(page)
    type = request.GET.get("type")
    keyword = request.GET.get("keyword")
    goods_list = []
    if type == 'byid':  # 按照商品id查
        if keyword:
            types = GoodsType.objects.get(id=int(keyword))
            goods_list = types.goods_set.order_by("goods_pro_date")
    elif type == 'bykey':  # 按商品名字查
        if keyword:
            goods_list = Goods.objects.filter(goods_name__contains=keyword).order_by("goods_pro_date")

    if goods_list:
        # 分页
        page_list = Paginator(goods_list, 15)
        goods_list = page_list.page(page)
        pages = page_list.page_range
        # 推荐商品
        nums = len(goods_list)
        nums = int(math.ceil(nums / 5))
        recommon_list = goods_list[:nums]

    return render(request, "buyer/goods_list.html", locals())


def good_detail(request, id):
    good = Goods.objects.filter(id=int(id)).first()
    return render(request, "buyer/detail.html", locals())


import math
import time
import datetime
from Buyer.models import *


@loginValid
def pay_order(request):
    """
    get请求 商品详情页购买单个商品。传入商品id，数量。
    post请求 购物车购买多个商品。
    """
    if request.method == "GET":
        num = request.GET.get("num")
        id = request.GET.get("id")

        if num and id:
            num = int(num)
            id = int(id)

            order = PayOrder()  # 订单
            order.order_number = str(time.time()).replace(".", "")
            order.order_date = datetime.datetime.now()
            order.order_user = LoginUser.objects.get(id=int(request.COOKIES.get("user_id")))

            order.save()

            good = Goods.objects.get(id=id)

            order_info = OrderInfo()  # 订单详情
            order_info.order_id = order
            order_info.goods_id = good.id
            order_info.goods_picture = good.goods_picture
            order_info.goods_name = good.goods_name
            order_info.goods_count = num
            order_info.goods_price = good.goods_price
            order_info.goods_total_price = round(good.goods_price * num, 3)
            order_info.store_id = good.goods_store

            order_info.order_status = 0  # 状态

            order_info.save()
            order.order_total = order_info.goods_total_price

            order.save()
    elif request.method == "POST":
        request_data = []
        data = request.POST
        data_item = request.POST.items()
        for key, value in data_item:
            if key.startswith("check_"):
                id = int(key.split("_", 1)[1])
                num = int(data.get("count_" + str(id)))
                request_data.append((id, num))
        if request_data:
            order = PayOrder()  # 创建订单
            order.order_number = str(time.time()).replace(".", "")
            order.order_date = datetime.datetime.now()

            order.order_user = LoginUser.objects.get(id=int(request.COOKIES.get("user_id")))

            order.order_total = 0.0
            order.goods_number = 0
            order.save()

            for id, num in request_data:
                good = Goods.objects.get(id=id)
                order_info = OrderInfo()  # 订单详情
                order_info.order_id = order
                order_info.goods_id = good.id
                order_info.goods_picture = good.goods_picture
                order_info.goods_name = good.goods_name
                order_info.goods_count = num
                order_info.goods_price = good.goods_price
                order_info.goods_total_price = round(good.goods_price * num, 3)
                order_info.store_id = good.goods_store

                order_info.order_status = 0

                order_info.save()

                order.order_total += order_info.goods_total_price  # 订单总价
                order.goods_number += 1  # 商品种类个数

            order.save()

    return render(request, "buyer/place_order.html", locals())


@loginValid
def alipayOrder(request):
    """
    阿里支付，传入交易订单号，总金额
    """
    order_number = request.GET.get("order_number")
    total = request.GET.get("total")

    # 实例化支付

    alipay = AliPay(
        appid="2016101200667714",
        app_notify_url=None,
        app_private_key_string=alipay_private_key_string,
        alipay_public_key_string=alipay_public_key_string,
        sign_type="RSA2"
    )

    order_string = alipay.api_alipay_trade_page_pay(
        out_trade_no=order_number,  # 订单编号
        total_amount=str(total),  # 金额 字符串类型
        subject="生鲜交易",
        return_url="http://127.0.0.1:8000/Buyer/pay_result/",  # 支付跳转页面
        notify_url="http://127.0.0.1:8000/Buyer/pay_result/",
    )

    result = "https://openapi.alipaydev.com/gateway.do?" + order_string

    return HttpResponseRedirect(result)


@loginValid
def pay_result(request):
    """
    支付结果页
    如果有out_trade_no，支付成功，修改订单状态
    """
    out_trade_no = request.GET.get("out_trade_no")
    if out_trade_no:
        payorder = PayOrder.objects.get(order_number=out_trade_no)
        payorder.orderinfo_set.all().update(order_status=1)

    return render(request, "buyer/pay_result.html", locals())
@loginValid
def delgood(request):
    sendData = {
        "code": 200,
        "data": ""
    }
    id = request.GET.get("id")
    if id:
        cart = Cart.objects.get(id=id)
        cart.delete()
        sendData["data"] = "删除编号%s成功"%id

    return JsonResponse(sendData)

@loginValid
def add_cart(request):
    """
    处理ajax 请求，添加商品到购物车 ，成功保存到数据库。
    传入商品id，数量
    """
    sendData = {
        "code": 200,
        "data": ""
    }
    if request.method == "POST":
        id = int(request.POST.get("goods_id"))
        count = int(request.POST.get("count", 1))

        goods = Goods.objects.get(id=id)
        cart = Cart()
        cart.goods_name = goods.goods_name
        cart.goods_num = count
        cart.goods_price = goods.goods_price
        cart.goods_picture = goods.goods_picture
        cart.goods_total = round(goods.goods_price * count, 3)
        cart.goods_id = goods.id
        cart.cart_user = request.COOKIES.get("user_id")
        cart.save()
        sendData['data'] = "加入购物车成功"
    else:
        sendData["code"] = 500
        sendData["data"] = "请求方式错误"
    return JsonResponse(sendData)


@loginValid
def mycart(request):
    id = request.COOKIES.get("user_id")
    carts = Cart.objects.filter(cart_user=id).order_by("-id")
    number = carts.count()
    return render(request, "buyer/cart.html", locals())


