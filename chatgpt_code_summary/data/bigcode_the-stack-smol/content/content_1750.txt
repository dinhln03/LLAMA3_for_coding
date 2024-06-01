#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================================================================
# Author Eduard Kabrinskyi <soulroot@gmail.com> Skype: soulroot@hotmail.com
# =========================================================================

# =========================
# Main APP definitions
# =========================
import logging
import os
import requests
from lxml import html
import time
from random import choice
# =========================
# Database APP definitions
# =========================
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.orm import Session
from sqlalchemy import func
# =========================
# Set Logging
# =========================
logging.basicConfig(format='%(asctime)s %(levelname)-7s %(module)s.%(funcName)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
logging.disable(logging.NOTSET)
logging.info('Loading %s', __name__)
# =========================
# Database Class
# =========================
Base = declarative_base()

class OrgTable(Base):
    __tablename__ = 'organization'
    id = Column(Integer, primary_key=True)
    name = Column(String(2000))
    inn = Column(Integer)
    address = Column(String(2000))

    def __init__(self, name, inn, address):
        self.name = name
        self.inn = inn
        self.address = address

    def __repr__(self):
        return "<Data %s, %s>" % (self.name, self.innm, self.address)

# =========================
# Spider Class
# =========================
class Busgov(object):

    def __init__(self):
        basename = 'database.db'
        self.engine = create_engine("sqlite:///%s" % basename, echo=False)
        if not os.path.exists(basename):
            Base.metadata.create_all(self.engine)
        f = open('page.txt', 'r')
        self.start = int(f.read())
        f.close()
        self.last_page = set()

    def get_count_items(self):
        self.session = Session(bind=self.engine)
        items = self.session.query(func.count(OrgTable.id)).scalar()
        self.session.close()
        return logging.info('Now Database items count: %s' %items)

    def get_pages(self, stop):
        try:
            for page in range(self.start, stop):
                logging.info('Crawl page: %s' % (page))
                page_text = get_page('http://bus.gov.ru/public/agency/choose.html?d-442831-p=' + str(page))
                tree = html.fromstring(page_text)
                org_list = tree.xpath('//table[@id="resultTable"]/tbody/tr[*]')
                x=1
                for org in org_list:
                    name = tree.xpath('//table[@id="resultTable"]/tbody/tr[' + str(x) + ']/td[2]/text()')[0].strip('\n  ')
                    inn = tree.xpath('//table[@id="resultTable"]/tbody/tr['+str(x)+']/td[3]/text()')[0]
                    address = tree.xpath('//table[@id="resultTable"]/tbody/tr['+str(x)+']/td[4]/text()')[0].strip('\n  ')
                    item = {'name': name, 'inn': inn, 'address': address}
                    x+=1
                    self.processed(item=item, page=page)
                f = open('page.txt', 'w')
                f.write(str(page))
                f.close()
            else:
                raise logging.error('Stop Crawl last page: %' % page)
        except Exception as e:
            logging.error(e.message)

    def processed(self, item, page):
        self.session = Session(bind=self.engine)
        #print item['name']
        ot = OrgTable(item['name'], item['inn'], item['address'])
        self.session.add(ot)
        self.session.commit()
        self.session.close()

# =========================
# Helper functions
# =========================
from requests.auth import HTTPDigestAuth, HTTPBasicAuth

proxies = {"http": (choice(list(open('proxy.txt')))).strip('\n')}


def get_request(page,proxies):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
        }
        r = requests.get(page, headers=headers, proxies=proxies, timeout=10.0)
        return r
    except:
        class r(object):
            status_code = None
        return r
        pass

    
def get_page(page):
    proxy_status = False
    sleep_time = (1)
    while proxy_status == False:
        time.sleep(sleep_time)
        logging.info("Set proxy: %s" %proxies["http"])
        r = get_request(page=page,proxies=proxies)
        if r.status_code == 200:
            proxy_status = True
            logging.info('Proxy UP: %s ' % proxies['http'])
        else:
            logging.info('Proxy DOWN: %s ' % proxies['http'])
            global proxies
            proxies = {"http": (choice(list(open('proxy.txt')))).strip('\n')}
    return r.text
# =========================
# bg.get_pages(xxxx) количество страниц всего
# в файле page.txt текущая страница с которой стартовать
# =========================
if __name__ == "__main__":
    bg = Busgov()
    bg.get_count_items()
    bg.get_pages(22278)
