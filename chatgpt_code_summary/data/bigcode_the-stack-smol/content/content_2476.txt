# -*- coding: utf-8 -*-
import scrapy

class scrapyshkmbab39Spider(scrapy.Spider):
    name = "scrapyshkmbab39"
    allowed_domains = ["ganjoor.net"]
    if 1 == 1:
        start_urls = ["https://ganjoor.net/hojviri/kashfol-mahjoob/kmbab39/sh"]
    else:
        start_urls = ["https://ganjoor.net/hojviri/kashfol-mahjoob/kmbab39/sh" + "1"]
    order = 1

    def parse(self, response):
        index = 0
        sh = dict()
        sh["type"] = "fasl"
        sh["text"] = dict()
        for i, poem in enumerate(response.css("div.poem>article>*")):
            if index == 0:
                if 0 == 1:
                    sh["title"] = "فصل" + " شماره " + str(self.order) + " - " + ''.join(poem.css("div.m1>p::text").extract()).strip()
                elif 0 == 2:
                    sh["title"] = "فصل" + " شماره " + str(self.order) + " - " + ''.join(poem.css("div.m2>p::text").extract()).strip()
                elif 0 == 3:
                    sh["title"] = "فصل" + " شماره " + str(self.order) + " - " + ''.join(response.css("div.poem>article>h2>a::text").extract()).strip() + ': ' + ''.join(poem.css("div.m1>p::text").extract()).strip()
                elif 0 == 4:
                    sh["title"] = "فصل" + " شماره " + str(self.order) + " - " + ''.join(response.css("div.poem>article>h2>a::text").extract()).strip() + ': ' + ''.join(poem.css("div.m2>p::text").extract()).strip()
                else:
                    sh["title"] = ''.join(response.css("div.poem>article>h2>a::text").extract_first()).strip()
            if poem.css("p::text").extract_first() is None or 'rel="bookmark"' in poem.css('*').extract_first() or 'class="spacer"' in poem.css('*').extract_first() or '<div style=' in poem.css('*').extract_first():
                continue
            if len(poem.css("div.m1>p")) == 1:
                if poem.css("div.b"):
                    if '٭٭٭' not in poem.css("div.m1>p::text").extract_first() and ''.join(poem.css("div.m1>p::text").extract()).strip() != '':
                        sh["text"][index] = dict([
                            ("m1", ''.join(poem.css("div.m1>p::text").extract()).strip()),
                            ("m2", ''.join(poem.css("div.m2>p::text").extract()).strip()),
                        ])
                else:
                    if '٭٭٭' not in poem.css("p:first-child::text").extract_first() and ''.join(poem.css("p:first-child::text").extract()).strip() != '':
                        sh["text"][index] = dict([
                            ("t1", ''.join(poem.css("p:first-child::text").extract()).strip()),
                            ("t2", ''.join(poem.css("p:last-child::text").extract()).strip()),
                        ])
            else:
                if poem.css("div.b2"):
                    if '٭٭٭' not in poem.css("p:first-child::text").extract_first() and ''.join(poem.css("p:first-child::text").extract()).strip() != '':
                        sh["text"][index] = dict([
                            ("t1", ''.join(poem.css("p:first-child::text").extract()).strip()),
                            ("t2", ''.join(poem.css("p:last-child::text").extract()).strip()),
                        ])
                else:
                    if '٭٭٭' not in poem.css('p::text').extract_first() and ''.join(poem.css('p::text').extract()).strip() != '':
                        sh['text'][index] = dict([
                            ('p', ''.join(poem.css('p::text').extract()).strip())
                        ])
            index = index + 1

        sh["order"] = self.order
        self.order = self.order + 1
        yield sh
        # next_page = response.css("div.navigation>div.navleft>a::attr(href)").extract_first()
        if self.order < (1 + 1):
            next_page = response.urljoin("https://ganjoor.net/hojviri/kashfol-mahjoob/kmbab39/sh" + str(self.order))
            yield scrapy.Request(next_page, callback=self.parse)
            