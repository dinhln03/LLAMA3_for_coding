import json
from decimal import Decimal

from django.core.paginator import Paginator
from django.db import transaction
from django.http import HttpResponseForbidden, JsonResponse
from django.shortcuts import render

# Create your views here.
from django.utils import timezone
from django.views import View
from django_redis import get_redis_connection

from goods.models import SKU
from meiduo_mall.utils.response_code import RETCODE
from meiduo_mall.utils.views import LoginRequiredMixin
from orders.models import OrderInfo, OrderGoods
from users.models import Address
import logging
logger = logging.getLogger('django')

class OrderSettlementView(LoginRequiredMixin,View):
    """结算订单"""

    def get(self,request):
        """提供订单结算页面"""
        # 获取登录用户
        user = request.user

        # 查询地址信息
        try:
            addresses = Address.objects.filter(user=user,is_deleted=False)

        except Address.DoesNotExist:
            # 如果地址为空,渲染模板时会判断,并跳转到地址编辑页面
            addresses = None


        # 从redis购物车中查询被勾选的商品信息
        redis_conn = get_redis_connection('carts')
        item_dict = redis_conn.hgetall('carts_%s' % user.id)
        cart_selected = redis_conn.smembers('selected_%s' % user.id)

        cart = {}

        for sku_id in cart_selected:
            cart[int(sku_id)] = int(item_dict[sku_id])


        # 准备初始值
        total_count = 0
        total_amount = Decimal(0.00)

        # 查询商品信息
        skus = SKU.objects.filter(id__in=cart.keys())

        for sku in skus:
            sku.count = cart[sku.id]
            sku.amount = sku.count * sku.price

            # 计算总数量和总金额
            total_count += sku.count
            total_amount += sku.amount

        # 补充运费
        freight = Decimal('10.00')

        # 渲染界面
        context = {
            'addresses':addresses,
            'skus':skus,
            'total_count': total_count,
            'total_amount': total_amount,
            'freight': freight,
            'payment_amount': total_amount + freight
        }

        return render(request,'place_order.html',context)


    def post(self,request):
        """保存订单信息和订单商品信息"""

        # 获取当前要保存的订单数据
        json_dict = json.loads(request.body)
        address_id = json_dict.get('address_id')
        pay_method = json_dict.get('pay_method')

        # 校验参数
        if not all([address_id, pay_method]):
            return HttpResponseForbidden('缺少必传参数')
            # 判断address_id是否合法
        try:
            address = Address.objects.get(id=address_id)
        except Exception:
            return HttpResponseForbidden('参数address_id错误')
            # 判断pay_method是否合法
        if pay_method not in [OrderInfo.PAY_METHODS_ENUM['CASH'],
                              OrderInfo.PAY_METHODS_ENUM['ALIPAY']]:
            return HttpResponseForbidden('参数pay_method错误')


        # 获取登录用户
        user = request.user

        # 生成订单编号:年月日时分秒+用户编号
        order_id = timezone.localtime().strftime('%Y%m%d%H%M%S') + ('%09d' % user.id)

        # 显式的开启一个事务
        with transaction.atomic():
            # 创建事务保存点
            save_id = transaction.savepoint()

            # 暴力回滚
            try:
                # 保存订单基本信息OrderInfo
                order = OrderInfo.objects.create(
                    order_id = order_id,
                    user = user,
                    address = address,
                    total_count=0,
                    total_amount=Decimal('0'),
                    freight=Decimal('10.00'),
                    pay_method=pay_method,
                    status=OrderInfo.ORDER_STATUS_ENUM['UNPAID'] if pay_method == OrderInfo.PAY_METHODS_ENUM['ALIPAY'] else OrderInfo.ORDER_STATUS_ENUM['UNSEND']
                )



                # 从redis读取购物车中被勾选的商品
                redis_conn = get_redis_connection('carts')
                item_dict = redis_conn.hgetall('carts_%s' % user.id)
                cart_selected = redis_conn.smembers('selected_%s' % user.id)

                carts = {}

                for sku_id in cart_selected:
                    carts[int(sku_id)] = int(item_dict[sku_id])

                # 获取选中的商品id
                sku_ids = carts.keys()

                # 遍历购物车中被勾选的商品信息
                for sku_id in sku_ids:

                    # TODO1: 增加的代码:增加一个死循环
                    while True:
                        # 查询SKU信息
                        sku = SKU.objects.get(id=sku_id)

                        # TODO2: 增加的代码:读取原始库存
                        origin_stock = sku.stock
                        origin_sales = sku.sales

                        # 判断SKU库存
                        sku_count = carts[sku_id]
                        # if sku_count > sku.stock:
                        if sku_count > origin_stock:
                            # 事务回滚
                            transaction.savepoint_rollback(save_id)
                            return JsonResponse({'code': RETCODE.STOCKERR,
                                                      'errmsg': '库存不足'})

                        # SKU减少库存,增加销量
                        # sku.stock -= sku_count
                        # sku.sales += sku_count
                        # sku.save()

                        # TODO3: 增加的代码:乐观锁更新库存和销量
                        # 计算差值
                        new_stock = origin_stock - sku_count
                        new_sales = origin_sales + sku_count

                        result = SKU.objects.filter(id=sku_id,stock=origin_stock).update(stock=new_stock,sales=new_sales)

                        #如果下单失败,但是库存足够时,继续下单,直到下单成功或者库存不足为止
                        if result == 0:
                            # 跳过当前循环的剩余语句，然后继续进行下一轮循环
                            continue

                        # 修改SPU销量
                        sku.goods.sales += sku_count
                        sku.goods.save()

                        # 保存订单商品信息OrderGoods
                        OrderGoods.objects.create(
                            order = order,
                            sku = sku,
                            count = sku_count,
                            price = sku.price
                        )

                        # 保存商品订单中总价和总数量
                        order.total_count += sku_count
                        order.total_amount += (sku_count * sku.price)

                        # TODO4:增加的代码:
                        # 下单成功或者失败就跳出循环
                        break

                # 添加邮费和保存订单信息
                order.total_amount += order.freight
                order.save()
            except Exception as e:
                logger.error(e)
                transaction.savepoint_rollback(save_id)
                return JsonResponse({
                    'code': RETCODE.DBERR,
                    'errmsg': '下单失败'})

            # 提交订单成功,显式的提交一次事务
            transaction.savepoint_commit(save_id)

        # 清楚购物车中已结算的商品
        pl = redis_conn.pipeline()
        pl.hdel('carts_%s' % user.id,*cart_selected)
        pl.srem('selected_%s' % user.id,*cart_selected)
        pl.execute()


        # 响应提交订单结果
        return JsonResponse({'code': RETCODE.OK,
                             'errmsg': '下单成功',
                             'order_id': order.order_id})



class OrderSuccessView(LoginRequiredMixin,View):
    def get(self, request):
        order_id = request.GET.get('order_id')
        payment_amount = request.GET.get('payment_amount')
        pay_method = request.GET.get('pay_method')

        context = {
            'order_id': order_id,
            'payment_amount': payment_amount,
            'pay_method': pay_method
        }
        return render(request, 'order_success.html', context)


class UserOrderInfoView(LoginRequiredMixin,View):

    def get(self,request,page_num):

        # 1.获取所有的订单
        orders = OrderInfo.objects.filter(user=request.user).order_by('-create_time')
        # 2.遍历获取每一个订单
        for order in orders:
            # 3.给每个订单绑定属性:status_name,pay_method_name,sku_list
            order.status_name = OrderInfo.ORDER_STATUS_CHOICES[order.status-1][1]

            order.pay_method_name = OrderInfo.PAY_METHOD_CHOICES[order.pay_method-1][1]

            order.sku_list = []
            # 4.给sku_list赋值 往里家sku
            # 5.获取订单商品所有对象
            lines = order.skus.all()
            # 6.遍历每一个订单商品,获取具体商品对象(sku表)
            for line in lines:
                sku = line.sku
                # 7.给商品绑定count,amount
                sku.count = line.count
                sku.amount = sku.price * sku.count

                # 8.给sku_list赋值
                order.sku_list.append(sku)

        # 9.调用分页器前将page_num转为整型
        page_num = int(page_num)
        try:
            # 10.生成一个分页器对象(orders是一个对象列表,2是每页显示的数量)
            paginator = Paginator(orders, 2)
            # 取某一页 返回某一页的所有对象
            page_orders = paginator.page(page_num)
            # 总页数
            total_page = paginator.num_pages
        except Exception as e:
            return HttpResponseForbidden('分页失败')

        # 11.拼接参数
        context = {
            'total_page':total_page,
            'page_orders':page_orders,
            'page_num':page_num
        }
        # 12.返回
        return render(request,'user_center_order.html',context)