import logging
import requests
import moment
import utils
import time
import json


class weather(utils.utils):
    def message_callback(self,ch, method, properties, body):
        logging.info('messgae received weather')
        time.sleep(1)
        self.__get_weather(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)  # 执行完再ack消息
        logging.info('weather messgae is ack on :'+str(moment.now()))

    def __get_weather(self,body):
        params = self.get_params(body)
        req = requests.get(params.request_url)
        data = json.loads(req.text)
        if(data['status'] == 200):
            wea = data['data']['forecast'][0]
            params.content='天气：'+wea['type']+' '+wea['high']+','+wea['low']+'   '+wea['notice']
        else:
            params.content='错误'

        self.notification(params)
        self.notification_to_zhouyu(params)

    def notification_to_zhouyu(self,params):
        url='http://push.devzhou.t.cn/Push/'+params.title+'/'+params.content+'?url='+params.red_url
        requests.get(url)
        logging.info('messgae send successfully')

