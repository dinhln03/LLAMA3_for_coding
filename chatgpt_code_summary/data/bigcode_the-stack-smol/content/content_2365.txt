import os

import click
from flask import Flask, render_template
from flask_wtf.csrf import CSRFError

from telechat.extensions import db, login_manager, csrf, moment
from telechat.blueprints.auth import auth_bp
from telechat.blueprints.chat import chat_bp
from telechat.blueprints.admin import admin_bp
from telechat.blueprints.oauth import oauth_bp
from telechat.settings import config
from telechat.models import User, Message


def register_extensions(app: Flask):
    """注册需要的扩展程序包到 Flask 程序实例 app 中"""
    db.init_app(app)  # 数据库 ORM
    login_manager.init_app(app)  # 登录状态管理
    csrf.init_app(app)  # CSRF 令牌管理
    moment.init_app(app)  # 时间格式化管理


def register_blueprints(app: Flask):
    """注册需要的蓝图程序包到 Flask 程序实例 app 中"""
    app.register_blueprint(auth_bp)
    app.register_blueprint(oauth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)


def register_errors(app: Flask):
    """注册需要的错误处理程序包到 Flask 程序实例 app 中"""

    @app.errorhandler(400)  # Bad Request 客户端请求的语法错误，服务器无法理解
    def bad_request(e):
        return render_template('error.html', description=e.description, code=e.code), 400

    @app.errorhandler(404)  # Not Found 服务器无法根据客户端的请求找到资源（网页）
    def page_not_found(e):
        return render_template('error.html', description=e.description, code=e.code), 404

    @app.errorhandler(500)  # Internal Server Error	服务器内部错误，无法完成请求
    def internal_server_error(e):
        return render_template('error.html', description="服务器内部错误，无法完成请求！", code="500"), 500

    @app.errorhandler(CSRFError)  # CSRF 验证失败
    def csrf_error_handle(e):
        return render_template('error.html', description=e.description, code=e.code), 400


def register_commands(app: Flask):
    """注册需要的CLI命令程序包到 Flask 程序实例 app 中"""

    @app.cli.command()
    @click.option('--drop', is_flag=True, help="创建之前销毁数据库")
    def initdb(drop: bool):
        """初始化数据库结构"""
        if drop:
            # 确认删除
            pass
        pass

    @app.cli.command()
    @click.option('--num', default=300, help="消息数量，默认为300")
    def forge(num: int):
        """生成虚拟数据"""
        pass


def create_app(config_name=None):
    """程序工厂：创建 Flask 程序，加载配置，注册扩展、蓝图等程序包"""

    # 从环境变量载入配置环境名称
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'development')

    # 创建 Flask 程序实例，程序名称为 telechat
    app = Flask('telechat')

    # 载入相应的配置
    app.config.from_object(config[config_name])

    # 注册程序包
    register_extensions(app)  # 扩展
    register_blueprints(app)  # 蓝图
    register_errors(app)  # 错误处理
    register_commands(app)  # CLI命令

    # 返回已配置好的 Flask 程序实例
    return app
