# -*- coding: utf-8 -*-

import time

from pymongo import MongoClient

from config import MONGO_CONFIG

def get_current_time(format_str: str = '%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间，默认为 2020-01-01 00:00:00 格式
    :param format_str: 格式
    :return:
    """
    return time.strftime(format_str, time.localtime())

class MongoDb:

    def __init__(self):
        """初始化
        初始化 mongo db
        """
        mongo_uri = 'mongodb://%s:%s@%s:%s' % (
            MONGO_CONFIG['user'],
            MONGO_CONFIG['pwd'],
            MONGO_CONFIG['host'],
            MONGO_CONFIG['port'])
        self.mongo = MongoClient(mongo_uri)
        self.sogou_db = self.mongo['sogou_dev']
        self.sogou_search_col = self.sogou_db['sogou_search_results']
        # self.task_db = self.mongo['sogou_tast']

    def update_sogou_login_cookie(self, username, cookie):
        """
        更新搜狗微信登录 cookie 信息
        :param username:
        :param cookie:
        :return:
        """
        col = self.sogou_db['sogou_login_cookies']
        ctime = get_current_time()
        find_obj = {
            'nickname': username,
            'is_valid': 1,
        }

        login_item = col.find_one(find_obj)

        print(login_item)

        # 插入新数据
        if not login_item:
            cookie = 'DESC=0; %s' % cookie
            col.insert_one({
                'cookie': cookie,
                'nickname': username,
                'device': '0',
                'state': 'normal',
                'c_time': ctime,
                'm_time': ctime,
                'is_valid': 1,
                'failures': 0,
            })
            return

        # 更新原有数据
        cookie = 'DESC=%s; %s' % (login_item['device'], cookie)
        col.update_one(find_obj, {
            '$set': {
                'state': 'normal',
                'cookie': cookie,
                'c_time': ctime,
                'm_time': ctime,
                'failures': 0,
            }
        })

    def insert_sogou_search_result(self, result):
        """
        保存搜狗搜索信息
        :param results: 结果数组
        """
        ctime = get_current_time()

        find_obj = {
            'id': result['id'],
            'is_valid': 1
        }

        search_item = self.sogou_search_col.find_one(find_obj)

        print(search_item)
        
        new_result = result
        # 插入新数据
        if not search_item:
            new_result["c_time"] = ctime
            new_result["m_time"] = ctime
            new_result["is_valid"] = 1
            self.sogou_search_col.insert_one(new_result)
            return

        # 更新原有数据
        new_result["m_time"] = ctime
        self.sogou_search_col.update_one(find_obj, {
            '$set': new_result
        })