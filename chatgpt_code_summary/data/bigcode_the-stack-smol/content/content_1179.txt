"""
Host management app
"""
from django.urls import path
from .views import *


app_name = 'sys_inspect'

urlpatterns = [
    # 设备列表
    path('device/list', InspectDevInfoViews.as_view(), name='inspect_devices_list'),

    # 添加设备
    path('device/add', AddDevView.as_view(), name='inspect_devices_add'),

    # 删除设备
    path('device/delete', DeleteDevView.as_view(), name='inspect_device_delete'),

    # 编辑设备
    path('device/edit', EditDevInfoView.as_view(), name='inspect_device_edit'),

    # 任务列表
    path('content/list', ContentViews.as_view(), name='inspect_contents_list'),

    # 添加任务
    path('content/add', AddContView.as_view(), name='inspect_contents_add'),

    # 删除任务
    path('content/delete', DeleteContView.as_view(), name='inspect_contents_delete'),

]


