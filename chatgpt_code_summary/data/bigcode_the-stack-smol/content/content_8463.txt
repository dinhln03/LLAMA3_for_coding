# -*- encoding: utf-8 -*-
import json
import importlib
import os
import builtins
from multiprocessing import Process

from importlib.util import find_spec

__all__=['run']

def run():
    from utils.RedisHelper import RedisHelper

    _redis=RedisHelper()
    _redis.pubsub=_redis.conn.pubsub()
    _redis.pubsub.subscribe(_redis.sub_name)
    sub_message=next(_redis.pubsub.listen())
    
    print(sub_message)  #订阅消息 
    # on manage.py -e local
    # {'type': 'subscribe', 'pattern': None, 'channel': b'nlp_test_pub', 'data': 1}   
    # if "subscribe"!=sub_message['type'] or _redis.sub_name!=sub_message["channel"].decode('utf-8','ignore'):
    #     raise "sub error"

    for message in _redis.pubsub.listen():
        if "message"!=message['type'] or _redis.sub_name!=sub_message["channel"].decode('utf-8','ignore'):
            print('type erro')
            continue
        # 默认不会有错误
 
        message['data']=message['data'].decode('utf-8','ignore')
        try:
            data=json.loads(message['data'])
        except:
            # 打印日志 #
            print('json parse error',message)
            continue

        # 控制必要的字段
        if "type" not in data:
            continue

        ### 暂时只进行单项任务
        if "initialize" !=data["type"]:
            continue

        # 获取该任务唯一id
        if "uid_list" not in data["data"]:
            continue

        id_list = data["data"]["uid_list"]
        try:
            uid=_redis.conn.rpop(id_list).decode('utf-8','ignore')
        except:
            print("uid error",uid)
            continue
        if int(uid)>=data["data"]["sub_count"]:
            raise "uid Index exceeded" 
            ## 优化报错
            
        # os.environ['uid']=uid
        # print("initialize uid is ",uid)

        if find_spec('handlers.'+data.get("type",""),package='..'):
            handlers=importlib.import_module('handlers.'+data.get("type",""),package='..')
        else:
            continue
            raise "import error"
            # 优化容错 #
        
        # 优化
        # if hasattr(handlers,message.get("type","")):
        #     handlers=getattr(handlers,message.get("type",""))

        p=Process(target=handlers.run,args=[data,int(uid)])
        p.start()
        p.join()
        