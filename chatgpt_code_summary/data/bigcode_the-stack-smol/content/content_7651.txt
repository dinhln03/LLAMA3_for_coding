import scrapy
from scrapy.spiders import Rule, CrawlSpider
from scrapy.linkextractors import LinkExtractor
from sulekha.items import SulekhaItem, AddressItem, AddressItemLoader, SulekhaItemLoader

class SulekhaScrapy(CrawlSpider):
    name = 'sulekha'
    allowed_domains =['sulekha.com','yellowpages.sulekha.com']
    start_urls = ['http://yellowpages.sulekha.com/']
    rules = [\
        Rule(LinkExtractor(restrict_xpaths=["//div[@class='sc-belt']/div/ul/li/a"]),\
        'parse_next', follow=True),\
        Rule(LinkExtractor(restrict_xpaths=["//h3[@data-ypcid]/a[@itemprop='url']"]),'parse_details'),\
        Rule(LinkExtractor(restrict_xpaths=["//li[@class='next']/a"]),\
        'parse_next', follow=True),\
        ]

    def parse_details(self, response):
        self.logger.info('Parse item called on %s', response.url)

        loader = SulekhaItemLoader(item=SulekhaItem(), response=response)
        loader.add_xpath('category', '//div[@itemprop="breadcrumb"]/a[3]/text()');
        loader.add_xpath('name', '//span[@itemprop="name"]/text()');
        loader.add_xpath('phone', '//em[@itemprop="telephone"]/text()');
        loader.add_value('address', self.parse_address_item(response));
        loader.add_xpath('email', '//span[@itemprop="email"]/a/text()');
        loader.add_xpath('website', '//a[@id="websitelink"]/text()');
        loader.add_xpath('contact_preson', '//div[@class="profile-child"]/text()');
        loader.add_xpath('working_hours', '//time[@itemprop="openingHours"]/em/text()');
        return loader.load_item()

    def parse_address_item(self, response):
        address_loader = AddressItemLoader(item=AddressItem(), response=response)
        address_loader.add_xpath('street_address', '//span[@itemprop="streetAddress"]/text()');
        address_loader.add_xpath('address_locality', '//span[@itemprop="addressLocality"]/a/text()');
        address_loader.add_xpath('address_region', '//span[@itemprop="addressRegion"]/text()');
        address_loader.add_xpath('postal_code', '//span[@itemprop="postalCode"]/text()');
        address_loader.add_xpath('land_mark', '//span[@class="land-mark"]/text()');
        return address_loader.load_item()

    def parse_next(self, response):
        self.logger.info('Parse next called on %s', response.url)
        yield scrapy.Request(response.url, callback=self.parse_details)
        