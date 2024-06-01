#!/usr/bin/python
# -*- coding: UTF-8 -*-

from scrapy.spiders import Spider
from scrapy.spiders import Request
import json
from hexun.items import HexunItem
from utils.urlUtils import UrlUtils
from utils.dateTimeUtils import DateTimeUtils

class PPSpider(Spider):
    name = 'pp'
    urlTemplate = 'http://webftcn.hermes.hexun.com/shf/minute?code=DCEpp{0}&start={1}&number=225&t=1513835351321'
    start_urls = [

    ]
    allowed_domains = ['*.hexun.com']

    def start_requests(self):
        contractList = DateTimeUtils.getContractList()
        for contract in contractList:
            url = self.urlTemplate.format(contract, DateTimeUtils.getStartTime())
            yield Request(url=url, callback=self.parseItem)

    def parseItem(self, response):
        jsonData = json.loads(response.body_as_unicode().strip(';').strip('(').strip(')'))
        datas = jsonData['Data'][0]
        contractName = self.getContractName(response)
        for dataItem in datas:
            lldpeItem = HexunItem()
            lldpeItem['product'] = contractName
            lldpeItem['dateTime'] = dataItem[0]
            lldpeItem['price'] = dataItem[1]
            lldpeItem['amount'] = dataItem[2]
            lldpeItem['volumn'] = dataItem[3]
            lldpeItem['avePrice'] = dataItem[4]
            lldpeItem['openInterest'] = dataItem[5]
            yield lldpeItem

    def getContractName(self, response):
        code = UrlUtils.getQueryValue(response.url, 'code')[-4:]
        return self.name + code
