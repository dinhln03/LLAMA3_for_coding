# -*- coding: utf-8 -*-
TIME_OUT = 60

EXCEPT_FILE = ['test.py','login.py','mix.py']

class Api(object):
    login = "/api/users/login"
    user_info="/api/users/info"
    signin = "/api/users/sign/signIn"
    map = "/api/RedEnvelope/updateUserMap"
    find_redbag = "/api/RedEnvelope/findReds"
    get_redbag = "/api/redUser/getRed"
    test=  "/api/sys/testJson"