from os.path import join as pjoin

from scrapy.spiders import (
    Rule,
    CrawlSpider,
)
from scrapy import exceptions
from scrapy.linkextractors import LinkExtractor

from django.conf import settings
from django.core.cache import caches

import tldextract

from core.extractors import ck0tp

from crawler import items


lockin = caches['lock_in_task']

EOAIENT = settings.ENDPOINTS['ck0tp']

ENDPOINT = EOAIENT['ENDPOINT']
ENDPATH = EOAIENT['ENDPATH']
DIRECTIVE = EOAIENT['DIRECTIVE']
DIRECTIVES = EOAIENT['DIRECTIVES']


class Ck0tp(CrawlSpider):
    name = 'ck0tp'

    allowed_domains = [
        tldextract.extract(ENDPOINT).registered_domain
    ]
    start_urls = [
        pjoin(ENDPOINT, DIRECTIVE),
        ENDPOINT,
    ] + [
        pjoin(ENDPOINT, d)
        for d in DIRECTIVES
    ]

    rules = (
        Rule(
            LinkExtractor(allow=(r'{}/\d+/?$'.format(ENDPATH), )),
            callback='parse_video', follow=True
        ),
    )

    def __init__(self, *args, **kwargs):
        super(Ck0tp, self).__init__(*args, **kwargs)
        # unduplicate lock
        if not lockin.add(self.__class__.__name__, 'true', 60 * 60 * 24 * 5):
            raise exceptions.CloseSpider('already launched spider')

    def closed(self, *args, **kwargs):
        lockin.delete(self.__class__.__name__)

    def parse_video(self, response):
        vid = ck0tp.Video(response.url)
        return items.Entry(vid.info())
