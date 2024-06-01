# coding=utf-8
"""
PAT - the name of the current project.
main_portfolio_maker.py - the name of the new file which you specify in the New File
dialog box during the file creation.
Hossein - the login name of the current user.
8 / 8 / 18 - the current system date.
9: 14 AM - the current system time.
PyCharm - the name of the IDE in which the file will be created.
"""

from portfolio_maker.subscriber import create_subscription
from observer import Observer
from price_fetcher.config import PROJECT_ID
from price_fetcher.publisher import list_topics
from price_fetcher.config import TICKERS
import time
import datetime


if __name__ == '__main__':
    topics = list_topics(PROJECT_ID)
    topics = [str(topic).split('/')[-1][:-2] for topic in topics if 'simulator' in str(topic)]
    subscriptions = [create_subscription(PROJECT_ID, topic, 'live_writer_' + str(i))
                     for i, topic in enumerate(topics)]
    observer = Observer(tickers=['AAPL'], start_date=datetime.date(2018, 10, 18))
    observer.initiate()
    for i in range(len(topics)):
        observer.receive_messages(PROJECT_ID, 'live_writer_' + str(i))
    while True:
        # print('PRINTING!', observer.instruments)
        time.sleep(60)
