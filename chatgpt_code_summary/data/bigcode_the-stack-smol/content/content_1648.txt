#-*- coding: utf-8 -*-

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from climatespider.items import ClimatespiderItem
from scrapy.selector import Selector
from dateutil.parser import parse
import re
import datetime
from scrapy.exceptions import CloseSpider

def getyesterdaty():
    today_date = datetime.date.today()
    yesterday_date = today_date - datetime.timedelta(days=1)
    return yesterday_date.strftime('%Y/%m/%d')

class wugSpider(CrawlSpider):
    name = "WUGCrawlSpider_AO"
    #today_date = datetime.now().strftime('%Y/%m/%d')
    allowed_domains = ['www.wunderground.com']
    start_urls = [
        'https://www.wunderground.com/history/airport/ZBAA/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/54618/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZBTJ/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZBYN/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZSSS/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/50888/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/50136/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZYHB/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/50854/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZSOF/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZLXY/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/54602/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/VMMC/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/54401/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/58506/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZGHA/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZSHC/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZHHH/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/58606/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZGGG/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZGSZ/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/53798/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZYTL/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/airport/ZUUU/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/50774/{0}/DailyHistory.html'.format(getyesterdaty()),
        'https://www.wunderground.com/history/station/50949/{0}/DailyHistory.html'.format(getyesterdaty())
    ]

    def parse(self, response):
        sel = Selector(response)
        indexlist = list(map(lambda x: x.replace(' ','').replace('.',''),sel.xpath('//table[@id="obsTable"]/thead/tr/th/text()').extract()))
        date = re.match(r'.*(\d{4}\/\d{1,2}\/\d{1,2}).*', response.url).group(1)
        datatable = sel.xpath('//tr[@class="no-metars"]')
        # items = []
        for each in datatable:
            item = ClimatespiderItem()
            item['area'] = re.match(r'.*history/(.*)/2\d{3}/.*', response.url).group(1)
            # item['date'] = date
            if len(indexlist) == 13:
                item['the_date'] = date
                item['the_time'] = parse(each.xpath('td[1]/text()').extract()[0]).strftime('%H:%M')
                item['qx_Humidity'] = each.xpath('td[5]/text()').extract()[0]
                item['qx_WindDir'] = each.xpath('td[8]/text()').extract()[0]
                item['qx_Precip'] = each.xpath('td[11]/text()').extract()[0]
                item['qx_Events'] = each.xpath('td[12]/text()').extract()[0].strip()
                try:
                    item['qx_Condition'] = each.xpath('td[13]/text()').extract()[0]
                except Exception as e:
                    item['qx_Condition'] = ''
                try:
                    item['qx_Temp'] = each.xpath('td[2]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Temp'] = each.xpath('td[2]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_WindChill_HeatIndex'] = each.xpath('td[3]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_WindChill_HeatIndex'] = each.xpath('td[3]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_DewPoint'] = each.xpath('td[4]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_DewPoint'] = each.xpath('td[4]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_Pressure'] = each.xpath('td[6]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Pressure'] = each.xpath('td[6]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_Visibility'] = each.xpath('td[7]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Visibility'] = each.xpath('td[7]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_WindSpeed'] = each.xpath('td[9]/span[1]/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_WindSpeed'] = each.xpath('td[9]/text()').extract()[0].strip().replace('-','')
                try:
                    item['qx_GustSpeed'] = each.xpath('td[10]/span[1]/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_GustSpeed'] = each.xpath('td[10]/text()').extract()[0].strip().replace('-','')
                yield item
            else:
                item['the_date'] = date
                item['the_time'] = parse(each.xpath('td[1]/text()').extract()[0]).strftime('%H:%M')
                item['qx_Humidity'] = each.xpath('td[4]/text()').extract()[0]
                item['qx_WindDir'] = each.xpath('td[7]/text()').extract()[0]
                item['qx_Precip'] = each.xpath('td[10]/text()').extract()[0]
                item['qx_Events'] = each.xpath('td[11]/text()').extract()[0].strip()
                try:
                    item['qx_Condition'] = each.xpath('td[12]/text()').extract()[0]
                except Exception as e:
                    item['qx_Condition'] = ''
                try:
                    item['qx_Temp'] = each.xpath('td[2]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Temp'] = each.xpath('td[2]/text()').extract()[0].strip().replace('-','')
                # try:
                #     item['WindChill_HeatIndex'] = each.xpath('td[3]/span/span[@class="wx-value"]/text()').extract()[0]
                # except Exception as e:
                #     item['WindChill_HeatIndex'] = each.xpath('td[3]/text()').extract()[0].strip().replace('-', '')
                try:
                    item['qx_DewPoint'] = each.xpath('td[3]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_DewPoint'] = each.xpath('td[3]/text()').extract()[0].strip().replace('-', '')
                try:
                    item['qx_Pressure'] = each.xpath('td[5]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Pressure'] = each.xpath('td[5]/text()').extract()[0].strip().replace('-', '')
                try:
                    item['qx_Visibility'] = each.xpath('td[6]/span/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_Visibility'] = each.xpath('td[6]/text()').extract()[0].strip().replace('-', '')
                try:
                    item['qx_WindSpeed'] = each.xpath('td[8]/span[1]/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_WindSpeed'] = each.xpath('td[8]/text()').extract()[0].strip().replace('-', '')
                try:
                    item['qx_GustSpeed'] = each.xpath('td[9]/span[1]/span[@class="wx-value"]/text()').extract()[0]
                except Exception as e:
                    item['qx_GustSpeed'] = each.xpath('td[9]/text()').extract()[0].strip().replace('-', '')
                yield item
            # for index in range(len(indexlist)):

