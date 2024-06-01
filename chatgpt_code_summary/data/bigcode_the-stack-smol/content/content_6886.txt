import requests
from . import FeedSource, _request_headers

# pylint: disable=no-member
class WorldCoinIndex(FeedSource):  # Weighted average from WorldCoinIndex 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = getattr(self, 'timeout', 15)
        if not hasattr(self, 'api_key'):
            raise Exception("WorldCoinIndex FeedSource requires 'api_key'.")

    def _fetch(self):
        feed = {}
        for base in self.bases:
            url = "https://www.worldcoinindex.com/apiservice/v2getmarkets?key={apikey}&fiat={base}"
            response = requests.get(url=url.format(apikey=self.api_key, base=base),
                                    headers=_request_headers, timeout=self.timeout)
            result = response.json()['Markets']
            for market in result:
                for ticker in market:
                    (quote, returnedBase) = ticker['Label'].split('/')
                    if base == returnedBase and quote in self.quotes:
                        self.add_rate(feed, base, quote, ticker['Price'], ticker['Volume_24h'] / ticker['Price'])
        return feed
