from scrapy import Spider
from scrapy.spiders import CrawlSpider, Rule
from scrapy.selector import Selector
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.linkextractors import LinkExtractor
import scrapy
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError
from UW_Madison.items import UwMadisonItem


class Madison_courses( CrawlSpider ):

    name = 'uw_madison5'

    allowed_domains = ['wisc.edu']

    start_urls = [
            
            "http://guide.wisc.edu/courses/", 
 
 
 ]
    rules = (

            Rule( LinkExtractor( allow = ( r'ttp://guide.wisc.edu/courses/' )),
                
                callback = 'parse_httpbin',

                follow = True

                ),

            )

    '''
    def  start_requests( self ):

        for u in self.start_urls:

            yield scrapy.Request( u, callback = self.parse_httpbin,
                    errback = self.errback_httpbin,
                    dont_filter = True )

    '''

    def parse_httpbin( self, response ):

        #self.logger.info("Got successful response {}".format(response.url) )

        items = UwMadisonItem()

        course = response.css('span.courseblockcode::text').extract()
        #course = response.css('span.courseblockcode::text').extract_first()
        
        title = response.css('div.sc_sccoursedescs > div.courseblock > p.courseblocktitle > strong::text').extract()
        #title = response.css('div.sc_sccoursedescs > div.courseblock > p.courseblocktitle > strong::text').extract_first()

        unit = response.css('.courseblockcredits::text').extract()
        #unit = response.css('.courseblockcredits::text').extract_first()

        description =  response.css('.courseblockdesc::text').extract()
        #description =  response.css('.courseblockdesc::text').extract_first()

        prerequisites =  response.css('p.courseblockextra.noindent.clearfix > span.cbextra-data > .bubblelink::text').extract()
        #prerequisites = response.css('p.courseblockextra.noindent.clearfix > span.cbextra-data > .bubblelink::text').extract_first()

        items['course'] = course
        items['title'] = title
        items['unit'] = unit
        items['description'] = description
        items['prerequisites'] = prerequisites

        yield items


    '''
    def errback_httpbin( self, failure):

        # log all failures
        self.logger.error(repr(failure))

        # in case you want to do something special for some errors,
        # you may need the failure's type:

        if failure.check(HttpError):

            # These exception come from HttpError spider middleware
            # you can get the non-200 response
            response = failure.value.response
            self.logger.error("HttpError on %s", response.url )

        elif failure.check(DNSLookupError):

            # This is the original request
            request = failure.request
            self.logger.error('DNSLookupError on %s', request.url )

        elif failure.check(TimeoutError, TCPTimeOutError ):

            request = failure.request
            self.logger.error('TimeoutError on %s', request.url)

    '''







        
