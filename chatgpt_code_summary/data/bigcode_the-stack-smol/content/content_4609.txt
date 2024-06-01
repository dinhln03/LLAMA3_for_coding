#!/usr/bin/env python
# encoding: utf-8

"""
@author: zhanghe
@software: PyCharm
@file: rack.py
@time: 2018-04-06 18:22
"""

from __future__ import unicode_literals

from datetime import datetime

from flask import (
    request,
    flash,
    render_template,
    url_for,
    redirect,
    abort,
    jsonify,
    Blueprint,
)
from flask_babel import gettext as _
from flask_login import login_required

from app_backend import app
from app_backend import excel
from app_backend.api.inventory import count_inventory
from app_backend.api.rack import (
    get_rack_pagination,
    get_rack_row_by_id,
    add_rack,
    edit_rack,
    get_rack_choices,
    # rack_current_stats,
    # rack_former_stats,
)
from app_backend.api.rack import (
    get_rack_rows,
    # get_distinct_brand,
)
from app_backend.api.warehouse import (
    get_warehouse_choices,
)
from app_backend.forms.rack import (
    RackSearchForm,
    RackAddForm,
    RackEditForm,
)
from app_backend.models.model_bearing import Rack
from app_backend.permissions.rack import (
    permission_rack_section_add,
    permission_rack_section_search,
    permission_rack_section_export,
    permission_rack_section_get,
    permission_rack_section_edit,
    permission_rack_section_del,
)
from app_common.maps.default import DEFAULT_SEARCH_CHOICES_INT_OPTION
from app_common.maps.operations import OPERATION_EXPORT, OPERATION_DELETE
from app_common.maps.status_delete import (
    STATUS_DEL_OK,
    STATUS_DEL_NO)

# 定义蓝图
bp_rack = Blueprint('rack', __name__, url_prefix='/rack')

# 加载配置
DOCUMENT_INFO = app.config.get('DOCUMENT_INFO', {})
PER_PAGE_BACKEND = app.config.get('PER_PAGE_BACKEND', 20)
AJAX_SUCCESS_MSG = app.config.get('AJAX_SUCCESS_MSG', {'result': True})
AJAX_FAILURE_MSG = app.config.get('AJAX_FAILURE_MSG', {'result': False})


@bp_rack.route('/lists.html', methods=['GET', 'POST'])
@login_required
@permission_rack_section_search.require(http_exception=403)
def lists():
    """
    货架列表
    :return:
    """
    template_name = 'rack/lists.html'
    # 文档信息
    document_info = DOCUMENT_INFO.copy()
    document_info['TITLE'] = _('rack lists')

    # 搜索条件
    form = RackSearchForm(request.form)
    form.warehouse_id.choices = get_warehouse_choices()
    # app.logger.info('')

    search_condition = [
        Rack.status_delete == STATUS_DEL_NO,
    ]
    if request.method == 'POST':
        # 表单校验失败
        if not form.validate_on_submit():
            flash(_('Search Failure'), 'danger')
            # 单独处理csrf_token
            if hasattr(form, 'csrf_token') and getattr(form, 'csrf_token').errors:
                map(lambda x: flash(x, 'danger'), form.csrf_token.errors)
        else:
            if form.warehouse_id.data != DEFAULT_SEARCH_CHOICES_INT_OPTION:
                search_condition.append(Rack.warehouse_id == form.warehouse_id.data)
            if form.name.data:
                search_condition.append(Rack.name == form.name.data)
        # 处理导出
        if form.op.data == OPERATION_EXPORT:
            # 检查导出权限
            if not permission_rack_section_export.can():
                abort(403)
            column_names = Rack.__table__.columns.keys()
            query_sets = get_rack_rows(*search_condition)

            return excel.make_response_from_query_sets(
                query_sets=query_sets,
                column_names=column_names,
                file_type='csv',
                file_name='%s.csv' % _('rack lists')
            )
        # 批量删除
        if form.op.data == OPERATION_DELETE:
            # 检查删除权限
            if not permission_rack_section_del.can():
                abort(403)
            rack_ids = request.form.getlist('rack_id')
            permitted = True
            for rack_id in rack_ids:
                # 检查是否正在使用
                # 库存
                if count_inventory(**{'rack_id': rack_id, 'status_delete': STATUS_DEL_NO}):
                    ext_msg = _('Currently In Use')
                    flash(_('Del Failure, %(ext_msg)s', ext_msg=ext_msg), 'danger')
                    permitted = False
                    break
            if permitted:
                result_total = True
                for rack_id in rack_ids:
                    current_time = datetime.utcnow()
                    rack_data = {
                        'status_delete': STATUS_DEL_OK,
                        'delete_time': current_time,
                        'update_time': current_time,
                    }
                    result = edit_rack(rack_id, rack_data)
                    result_total = result_total and result
                if result_total:
                    flash(_('Del Success'), 'success')
                else:
                    flash(_('Del Failure'), 'danger')
    # 翻页数据
    pagination = get_rack_pagination(form.page.data, PER_PAGE_BACKEND, *search_condition)

    # 渲染模板
    return render_template(
        template_name,
        form=form,
        pagination=pagination,
        **document_info
    )


@bp_rack.route('/<int:rack_id>/info.html')
@login_required
@permission_rack_section_get.require(http_exception=403)
def info(rack_id):
    """
    货架详情
    :param rack_id:
    :return:
    """
    # 详情数据
    rack_info = get_rack_row_by_id(rack_id)
    # 检查资源是否存在
    if not rack_info:
        abort(404)
    # 检查资源是否删除
    if rack_info.status_delete == STATUS_DEL_OK:
        abort(410)
    # 文档信息
    document_info = DOCUMENT_INFO.copy()
    document_info['TITLE'] = _('rack info')
    # 渲染模板
    return render_template('rack/info.html', rack_info=rack_info, **document_info)


@bp_rack.route('/add.html', methods=['GET', 'POST'])
@login_required
@permission_rack_section_add.require(http_exception=403)
def add():
    """
    创建货架
    :return:
    """
    template_name = 'rack/add.html'
    # 文档信息
    document_info = DOCUMENT_INFO.copy()
    document_info['TITLE'] = _('rack add')

    # 加载创建表单
    form = RackAddForm(request.form)
    form.warehouse_id.choices = get_warehouse_choices(option_type='create')

    # 进入创建页面
    if request.method == 'GET':
        # 渲染页面
        return render_template(
            template_name,
            form=form,
            **document_info
        )

    # 处理创建请求
    if request.method == 'POST':
        # 表单校验失败
        if not form.validate_on_submit():
            flash(_('Add Failure'), 'danger')
            return render_template(
                template_name,
                form=form,
                **document_info
            )

        # 表单校验成功
        current_time = datetime.utcnow()
        rack_data = {
            'warehouse_id': form.warehouse_id.data,
            'name': form.name.data,
            'create_time': current_time,
            'update_time': current_time,
        }
        result = add_rack(rack_data)
        # 创建操作成功
        if result:
            flash(_('Add Success'), 'success')
            return redirect(request.args.get('next') or url_for('rack.lists'))
        # 创建操作失败
        else:
            flash(_('Add Failure'), 'danger')
            return render_template(
                template_name,
                form=form,
                **document_info
            )


@bp_rack.route('/<int:rack_id>/edit.html', methods=['GET', 'POST'])
@login_required
@permission_rack_section_edit.require(http_exception=403)
def edit(rack_id):
    """
    货架编辑
    """
    rack_info = get_rack_row_by_id(rack_id)
    # 检查资源是否存在
    if not rack_info:
        abort(404)
    # 检查资源是否删除
    if rack_info.status_delete == STATUS_DEL_OK:
        abort(410)

    template_name = 'rack/edit.html'

    # 加载编辑表单
    form = RackEditForm(request.form)
    form.warehouse_id.choices = get_warehouse_choices(option_type='update')

    # 文档信息
    document_info = DOCUMENT_INFO.copy()
    document_info['TITLE'] = _('rack edit')

    # 进入编辑页面
    if request.method == 'GET':
        # 表单赋值
        form.warehouse_id.data = rack_info.warehouse_id
        form.name.data = rack_info.name
        # form.create_time.data = rack_info.create_time
        # form.update_time.data = rack_info.update_time
        # 渲染页面
        return render_template(
            template_name,
            rack_id=rack_id,
            form=form,
            **document_info
        )

    # 处理编辑请求
    if request.method == 'POST':
        # 表单校验失败
        if not form.validate_on_submit():
            flash(_('Edit Failure'), 'danger')
            return render_template(
                template_name,
                rack_id=rack_id,
                form=form,
                **document_info
            )
        # 表单校验成功
        current_time = datetime.utcnow()
        rack_data = {
            'warehouse_id': form.warehouse_id.data,
            'name': form.name.data,
            'update_time': current_time,
        }
        result = edit_rack(rack_id, rack_data)
        # 编辑操作成功
        if result:
            flash(_('Edit Success'), 'success')
            return redirect(request.args.get('next') or url_for('rack.lists'))
        # 编辑操作失败
        else:
            flash(_('Edit Failure'), 'danger')
            return render_template(
                template_name,
                rack_id=rack_id,
                form=form,
                **document_info
            )


@bp_rack.route('/ajax/del', methods=['GET', 'POST'])
@login_required
def ajax_delete():
    """
    货架删除
    :return:
    """
    ajax_success_msg = AJAX_SUCCESS_MSG.copy()
    ajax_failure_msg = AJAX_FAILURE_MSG.copy()

    # 检查删除权限
    if not permission_rack_section_del.can():
        ext_msg = _('Permission Denied')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)

    # 检查请求方法
    if not (request.method == 'GET' and request.is_xhr):
        ext_msg = _('Method Not Allowed')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)

    # 检查请求参数
    rack_id = request.args.get('rack_id', 0, type=int)
    if not rack_id:
        ext_msg = _('ID does not exist')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)

    rack_info = get_rack_row_by_id(rack_id)
    # 检查资源是否存在
    if not rack_info:
        ext_msg = _('ID does not exist')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)
    # 检查资源是否删除
    if rack_info.status_delete == STATUS_DEL_OK:
        ext_msg = _('Already deleted')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)
    # 检查是否正在使用
    # 库存
    if count_inventory(**{'rack_id': rack_id, 'status_delete': STATUS_DEL_NO}):
        ext_msg = _('Currently In Use')
        ajax_failure_msg['msg'] = _('Del Failure, %(ext_msg)s', ext_msg=ext_msg)
        return jsonify(ajax_failure_msg)

    current_time = datetime.utcnow()
    rack_data = {
        'status_delete': STATUS_DEL_OK,
        'delete_time': current_time,
        'update_time': current_time,
    }
    result = edit_rack(rack_id, rack_data)
    if result:
        ajax_success_msg['msg'] = _('Del Success')
        return jsonify(ajax_success_msg)
    else:
        ajax_failure_msg['msg'] = _('Del Failure')
        return jsonify(ajax_failure_msg)


@bp_rack.route('/ajax/get_rack_choices', methods=['GET', 'POST'])
@login_required
def ajax_get_rack_choices():
    """
    货架选项
    :return:
    """
    warehouse_id = request.args.get('warehouse_id', 0, type=int)
    rack_choices = get_rack_choices(warehouse_id)
    return jsonify(rack_choices)

# @bp_rack.route('/ajax/stats', methods=['GET', 'POST'])
# @login_required
# def ajax_stats():
#     """
#     获取货架统计
#     :return:
#     """
#     time_based = request.args.get('time_based', 'hour')
#     result_rack_current = rack_current_stats(time_based)
#     result_rack_former = rack_former_stats(time_based)
#
#     line_chart_data = {
#         'labels': [label for label, _ in result_rack_current],
#         'datasets': [
#             {
#                 'label': '在职',
#                 'backgroundColor': 'rgba(220,220,220,0.5)',
#                 'borderColor': 'rgba(220,220,220,1)',
#                 'pointBackgroundColor': 'rgba(220,220,220,1)',
#                 'pointBorderColor': '#fff',
#                 'pointBorderWidth': 2,
#                 'data': [data for _, data in result_rack_current]
#             },
#             {
#                 'label': '离职',
#                 'backgroundColor': 'rgba(151,187,205,0.5)',
#                 'borderColor': 'rgba(151,187,205,1)',
#                 'pointBackgroundColor': 'rgba(151,187,205,1)',
#                 'pointBorderColor': '#fff',
#                 'pointBorderWidth': 2,
#                 'data': [data for _, data in result_rack_former]
#             }
#         ]
#     }
#     return json.dumps(line_chart_data, default=json_default)
#
#
# @bp_rack.route('/stats.html')
# @login_required
# @permission_rack_section_stats.require(http_exception=403)
# def stats():
#     """
#     货架统计
#     :return:
#     """
#     # 统计数据
#     time_based = request.args.get('time_based', 'hour')
#     if time_based not in ['hour', 'date', 'month']:
#         abort(404)
#     # 文档信息
#     document_info = DOCUMENT_INFO.copy()
#     document_info['TITLE'] = _('rack stats')
#     # 渲染模板
#     return render_template(
#         'rack/stats.html',
#         time_based=time_based,
#         **document_info
#     )
#
#
# @bp_rack.route('/<int:rack_id>/stats.html')
# @login_required
# @permission_rack_section_stats.require(http_exception=403)
# def stats_item(rack_id):
#     """
#     货架统计明细
#     :param rack_id:
#     :return:
#     """
#     rack_info = get_rack_row_by_id(rack_id)
#     # 检查资源是否存在
#     if not rack_info:
#         abort(404)
#     # 检查资源是否删除
#     if rack_info.status_delete == STATUS_DEL_OK:
#         abort(410)
#
#     # 统计数据
#     rack_stats_item_info = get_rack_row_by_id(rack_id)
#     # 文档信息
#     document_info = DOCUMENT_INFO.copy()
#     document_info['TITLE'] = _('rack stats item')
#     # 渲染模板
#     return render_template(
#         'rack/stats_item.html',
#         rack_stats_item_info=rack_stats_item_info,
#         **document_info
#     )
