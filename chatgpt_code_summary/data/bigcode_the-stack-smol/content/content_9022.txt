import scrapy
import json


class RussianAlphabetSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        'https://www.russianforeveryone.com/',
    ]

    def parse(self, response):
        alphabets_urls = response.xpath('//a[contains(.,"alphabet")]')
        yield from response.follow_all(alphabets_urls, callback=self.parse_alphabet)

        
    def parse_alphabet(self, response):
        tables = response.xpath('//table[contains(.,"Letter")]')
        rows = tables.xpath("*")
        alphabet_dict = {}
        i = 0
        for row in rows:
            alphabet_dict[i] = {
                "text": row.xpath("descendant::node()/text()[normalize-space()]").getall(),
                "image": row.xpath("descendant::node()/img[@src]/@src").getall(),
                "sound": row.xpath("descendant::node()/a/@onclick").getall(),
            }
            i += 1

        with open('banana_crawler/out/russian_alphabet.json', 'w') as f:
            json.dump(alphabet_dict, f, indent=4)

