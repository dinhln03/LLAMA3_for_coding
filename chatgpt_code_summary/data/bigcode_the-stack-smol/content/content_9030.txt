# coding=utf-8

import pywikibot
import re
import urllib.request
from pywikibot import Claim, pagegenerators as pg
from time import sleep
from urllib.error import HTTPError
from utils.properties import PID_OK_ACCOUNT, PID_OK_PROFILE_ID

repo = pywikibot.Site('wikidata', 'wikidata')


def parse_ok_ru(account):
    html = None
    tries = 0
    while html is None:
        try:
            html = urllib.request.urlopen('https://ok.ru/' + account).read().decode()
        except HTTPError:
            if tries > 3:
                return None
            tries += 1
            sleep(2 ** tries)
    match = re.search('<a data-module="AuthLoginPopup" href="/profile/(\\d+)"', html)
    if match is None:
        return None
    return match.group(1)


def process_claim(claim):
    if PID_OK_PROFILE_ID in claim.qualifiers:
        return

    account = claim.getTarget()

    match = re.fullmatch('^profile/(\\d+)$', account)
    if match:
        account_id = match.group(1)
    else:
        account_id = parse_ok_ru(account)

    if account_id is None:
        print("%s -> group" % account)
        return

    print("%s -> %s" % (account, account_id))
    qualifier = Claim(repo, PID_OK_PROFILE_ID)
    qualifier.setTarget(account_id)
    claim.addQualifier(qualifier)
    return


def add_ok_numeric_id(item):
    data = item.get()
    if 'claims' not in data or PID_OK_ACCOUNT not in data['claims']:
        return
    for claim in data['claims'][PID_OK_ACCOUNT]:
        process_claim(claim)


def iterate_items():
    query = '''
        SELECT ?item
        {
          ?item p:P5163 ?statement .
          ?statement ps:P5163 ?value .
          #?item wdt:P31 wd:Q5 .
          ?article schema:about ?item .
          ?article schema:isPartOf <https://ru.wikipedia.org/>.
          FILTER REGEX(?value, "^(?!group/)", "i")
          FILTER NOT EXISTS{ ?statement pq:P9269 [] }
        }
    '''
    generator = pg.WikidataSPARQLPageGenerator(query, site=repo)
    for item in generator:
        add_ok_numeric_id(item)
        sleep(5)


iterate_items()
