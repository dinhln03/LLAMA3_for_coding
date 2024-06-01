""" YQL out mkt cap and currency to fill out yahoo table """

""" TODO: retreive lists of 100 symbols from database and update"""
""" Results are intented to use while matching yahoo tickers, which one has mkt cap? which ones has sector? """

import mysql.connector
import stockretriever
import sys
import time
from random import randint

cnx = mysql.connector.connect(user='root', password='root', database='yahoo')
cursor = cnx.cursor()

sleeptime = 10

add_market_cap = ("INSERT INTO stocks "
                "(symbol, market_cap, currency) "
                "VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE market_cap=VALUES(market_cap), currency=VALUES(currency)")

 

get_new_symbols = """SELECT symbol
    FROM yahoo.stocks 
    WHERE market_cap is NULL
    and currency is NULL"""

try:
    cursor.execute(get_new_symbols)
except mysql.connector.errors.IntegrityError, e: 
    print(e) 


for result in cursor.fetchall(): 
    for symbol in result:
        data = []
        market_cap = ""
        currency = ""
        try:
            data = stockretriever.get_current_info([symbol])
        except TypeError as e:
            #print "Typerror {0}: {1}".format(e.errno, e.strerror)
            print "Type error, could not fetch current info on ", symbol

        except Exception as e:
            print(e)

        try:
            currency = data['Currency']
            market_cap = data['MarketCapitalization']
        except Exception as e:
            print "No currency or mkt cap error", e
            continue

        data_company = (symbol, market_cap, currency)
        try:
            cursor.execute(add_market_cap, data_company)
        except mysql.connector.errors.IntegrityError, e: 
            print(e)
            continue
        try:
            print "Success updating", symbol, currency, market_cap
        except UnicodeEncodeError as e:
            print e

        cnx.commit()
        time.sleep(randint(0,sleeptime))

cursor.close()
cnx.close()
