import logging

from urllib.parse import urljoin

import scrapy
from scrapy import Request

from scrapy_selenium import SeleniumRequest
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from sainsburys.items import SainsburysItem

logger = logging.getLogger('spam_application')
logger.setLevel(logging.DEBUG)


class BasicSpider(scrapy.Spider):
    name = 'basic'
    allowed_domains = ['www.sainsburys.co.uk']
    start_urls = ['https://www.sainsburys.co.uk/shop/gb/groceries/meat-fish']

    def parse(self, response):
        
        urls = response.css("ul.categories.departments li a::attr(href)").extract()

        for url in urls:
            
            yield response.follow(url, callback=self.parse_department)

    def parse_department(self, response):

        products = response.css("ul.productLister.gridView").extract()

        if products:

            for product in self.handle_product_listing(response):
                yield product
        
        pages = response.css("ul.categories.shelf li a::attr(href)").extract()

        if not pages:
            pages = response.css("ul.categories.aisles li a::attr(href)").extract()

        if not pages:
            return
        
        for url in pages:
            yield response.follow(url, callback=self.parse_department)

    def handle_product_listing(self, response):

        selector = 'button.ln-c-button'

        urls = response.css("ul.productLister.gridView li.gridItem h3 a::attr(href)").extract()

        for url in urls:
            # yield response.follow(url, callback=self.parse_product)
            yield SeleniumRequest(url=url, callback=self.parse_product, wait_time=10, wait_until=EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))

        next_page = response.css("#productLister > div.pagination.paginationBottom > ul > li.next > a::attr(href)").extract()

        if next_page:
            yield response.follow(next_page, callback=self.handle_product_listing)

    def parse_product(self, response):

        product_name = response.css("h1.pd__header::text").extract()[0]
        product_image = response.css("img.pd__image::attr(src)").extract()[0]

        item = SainsburysItem()

        item["url"] = response.url
        item["name"] = product_name
        item["image"] = product_image

        yield item
