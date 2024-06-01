#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import subprocess
import time

import web
from web import form,timelimit
import hashlib
import datetime
import random
from func_recall import risk_recall

_tmpstring = "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',20))
urls = (
    '/', 'Login',
    '/login', 'Login',
    '/base'+_tmpstring,'Datatable',
    '/data_set/(.*)/(.*)','download',
    '/pyrule'+_tmpstring,'Pyrule',
)

render = web.template.render('templates')
app = web.application(urls, globals())

# 密码表-risk#ok#
user_dict = {"root": "44dbb6a8dcf8471f8145bd3d7f8e30f0",
             "risk": "0b9fdd4bcbdddc099a5ce9ca7d855bb7",
             "recall": "bf83db0887c22e98590da5a3d18fcb15"
             }
login_validity = 3600*3    # 登陆有效期3小时

# 1.登陆表
class Login:
    def __init__(self):
        global recall_p
        if recall_p:
            try:
                recall_p.kill()
            except:
                pass
        curtime = datetime.datetime.now().strftime('%Y-%m-%d')
        random.seed(curtime)
        global today_passwd #每天的随机临时登陆密码
        today_passwd = "".join(random.sample('zyxwvutsrqponmlkjihgfedcba', 20))
        user_dict.update({'admin': str(hashlib.md5(today_passwd).hexdigest())})

        self.login_form = form.Form(
            form.Textbox("username", description="Username"),
            form.Password("password", description="Password"),
            form.Button("login", type="submit", description="Login")
        )

    def GET(self):
        return render.logins(self.login_form(),message="")

    def POST(self):
        param = web.input()
        user_name = param.username
        user_password = hashlib.md5(param.password).hexdigest()
        if ('recall' == user_name) & user_dict.has_key(user_name) & (user_password == user_dict.get(user_name)):
            global loan_time
            loan_time = datetime.datetime.now()
            raise web.seeother('/pyrule'+_tmpstring)
        if user_dict.has_key(user_name) & (user_password == user_dict.get(user_name)):
            loan_time = datetime.datetime.now()
            raise web.seeother('/base'+_tmpstring)
        else:
            message = "False password or User, please contact the administrator."
            return render.logins(self.login_form(),message)

# 2.数据信息表
class Datatable:
    def __init__(self):
        self.message_user = "user_name: admin, password_today: {}".format(today_passwd)
        self.dataset = {
           "okash_new_user":["/data_set/okash/new_user.xlsx","Okash首贷报表下载"],
           "opesa_new_user": ["/data_set/opesa/new_user.xlsx", "Opesa首贷报表下载"],
           "mhela_new_user": ["/data_set/mhela/new_user.xlsx", "Mhela首贷报表下载"],
           "okash_old_user": ["/data_set/okash/old_user.xlsx", "Okash复贷报表下载"],
           "opesa_old_user": ["/data_set/opesa/old_user.xlsx", "Opesa复贷报表下载"],
           "mhela_old_user": ["/data_set/mhela/old_user.xlsx", "Mhela复贷报表下载"]
        }

    def GET(self):
        duration_time = datetime.datetime.now()
        if (duration_time - loan_time).total_seconds()>login_validity: # 重新登录
            raise web.seeother('/login')
        else:
            message_user =  self.message_user
            form_submit,df_sheet = self.get_data()
            href_link, href_name = '',''
            return render.data_base(message_user,form_submit,df_sheet,href_link, href_name)

    def POST(self):
        duration_time = datetime.datetime.now()
        if (duration_time - loan_time).total_seconds() > login_validity:  # 重新登录
            raise web.seeother('/login')
        else:
            param = web.input()
            product_name = param.get('product_name')
            table_name = param.get('table_name')
            user_phases = param.get('user_phases', '')
            message_user = self.message_user + ",  current_table: {}.{}.{}".format(product_name, user_phases,table_name)
            form_submit, df_sheet = self.get_data(product_name, user_phases, table_name)
            href_link, href_name = self.dataset.get("{}_{}".format(product_name,user_phases),['',''])
            return render.data_base(message_user, form_submit,df_sheet,href_link, href_name)

    def get_data(self, product_name='okash', user_phases='new_user', table_name='table_describe'):
        _template = 'templates/{}/{}/{}.html'.format(product_name, user_phases, table_name)
        if os.path.exists(_template):
            _data = open(_template, 'r').read()
            _form = self.get_form(product_name, user_phases, table_name)
        else:
            _template = 'templates/{}/{}/sheet_summary.html'.format(product_name, user_phases)
            _data = open(_template, 'r').read()
            _form = self.get_form(product_name)
        return _form, _data

    def get_form(self, product_name='okash', user_phases='new_user', table_name='table_describe'):
        data_path = 'templates/{}/{}/'.format(product_name, user_phases)
        assert os.path.exists(data_path), 'the path of "{}" is error'.format(data_path)
        table_list = [i[:-5] for i in os.listdir(data_path) if i.startswith('df_') & i.endswith('.html')]
        table_list.sort()
        sort_dict = {'rules': 10, 'apply': 20, 'loan': 30, 'history': 40}
        table_list.sort(key=lambda x: sort_dict.get(x.split('_')[1], 100))
        form_submit = form.Form(
            form.Dropdown(name='product_name', args=['okash', 'opesa','mhela'], value=product_name),
            form.Dropdown(name='user_phases', args=['new_user', 'old_user'], value=user_phases),
            form.GroupedDropdown(name='table_name',
                                 args=(('INTRODUCTION', ['table_describe']), ('DATA INFO', table_list)),
                                 value=table_name),
            form.Button("Submit", value="submit", description="data_table")
        )
        return form_submit

recall_p = None #回调进程(后台抽取数据)
# 3.风控召回表
class Pyrule:
    def __init__(self):
        self.message_user = "风控端—用户精准召回"
        self.data_path = "./templates/recall/"
        self.dump_file = ""
        self.start_time = (datetime.datetime.now()+datetime.timedelta(-1)).strftime('%Y%m%d000000')
        self.end_time = datetime.datetime.now().strftime('%Y%m%d000000')
        self.from_product = 'okash'
        self.into_product = 'mhela'
        self.risk_rules = 'new'

    def GET(self):
        duration_time = datetime.datetime.now()
        if (duration_time - loan_time).total_seconds()>login_validity:  # 重新登录
            raise web.seeother('/login')
        else:
            message_user = self.message_user
            form_submit = self.get_form()
            df_sheet = ''
            href_link, href_name = '',''
            return render.data_base(message_user,form_submit, df_sheet, href_link, href_name)

    def POST(self):
        duration_time = datetime.datetime.now()
        if (duration_time - loan_time).total_seconds() > login_validity:  # 重新登录
            raise web.seeother('/login')
        else:
            param = web.input()
            self.from_product = param.get('from_product')
            self.into_product = param.get('into_product')
            self.risk_rules = param.get('risk_rules', '')
            self.start_time = param.get('start_time')
            self.end_time = param.get('end_time')
            self.dump_file = '{}recall_{}'.format(self.data_path,self.from_product)

            form_submit = self.get_form()
            if param.get('Submit'):
                global recall_p
                if recall_p:
                    try:
                        recall_p.kill()
                    except:
                        pass
                message_user = self.message_user + ": {} -> {}.{}  ({}-{},查询中。。。)".format(
                    self.from_product, self.into_product, self.risk_rules, self.start_time, self.end_time)

                recall_p = risk_recall(self.from_product, self.into_product, self.risk_rules, self.start_time,
                        self.end_time, self.dump_file)  # 更新数据
                return render.data_base(message_user, form_submit, "", "", "")
            if param.get('Refresh'):
                if recall_p:
                    message_user = self.message_user + ": {} -> {}.{}  ({}-{},查询中。。。)".format(
                        self.from_product, self.into_product, self.risk_rules, self.start_time, self.end_time)
                    status = subprocess.Popen.poll(recall_p)
                    if 0 == status:
                        df_sheet,len_data = self.get_data()
                        message_user = self.message_user + ": {} -> {}.{}  ({}-{},共计{}条)".format(
                            self.from_product, self.into_product, self.risk_rules, self.start_time, self.end_time, len_data)
                        href_link = '/data_set/recall/recall_{}.xlsx'.format(self.from_product)
                        href_name = "recall_{}下载".format(self.from_product)
                        return render.data_base(message_user, form_submit, df_sheet,href_link, href_name)
                    else:
                        return render.data_base(message_user, form_submit, "", "", "")
                return render.data_base(self.message_user, form_submit, "", "", "")

    def get_data(self):
        _template = '{}.html'.format(self.dump_file)
        if os.path.exists(_template):
            _data = open(_template, 'r').read()
        else:
            _data = ''
        _text = '{}.txt'.format(self.dump_file)
        if os.path.exists(_text):
            _txt = open(_text, 'r').read()
        else:
            _txt = ' '
        return _data,_txt

    def get_form(self):
        form_submit = form.Form(
            form.Dropdown(name='from_product', args=['okash', 'opesa', 'mhela'], value=self.from_product),
            form.Dropdown(name='into_product', args=['okash', 'opesa', 'mhela'], value=self.into_product),
            form.Dropdown(name='risk_rules', args=['new', 'old'], value=self.risk_rules),
            form.Input(name='start_time', type="search", min='20170101000000', max='20250101000000', value=self.start_time),
            form.Input(name='end_time', type="search", min='20170101000000', max='20250101000000', value=self.end_time),
            form.Button("Submit", value="submit", description="data_table"),
            form.Button("Refresh", value="submit", description="data_table")
        )
        return form_submit

BUF_SIZE = 262144
class download:
    def GET(self,file_name,phases):
        file_path = os.path.join('templates', file_name,phases)
        f = None
        try:
            f = open(file_path, "rb")
            web.header('Content-Type','application/octet-stream')
            web.header('Content-disposition', 'attachment; filename=%s.xlsx' % file_name)
            while True:
                c = f.read(BUF_SIZE)
                if c:
                    yield c
                else:
                    break
        except Exception, e:
            print e
            yield 'Error'
        finally:
            if f:
                f.close()

if __name__ == '__main__':
    app.run()
