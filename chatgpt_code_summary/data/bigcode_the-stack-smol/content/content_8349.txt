import gdax
import os
import json

API_KEY = os.environ['GDAX_API_KEY']
API_SECRET = os.environ['GDAX_API_SECRET']
API_PASS = os.environ['GDAX_API_PASS']


def main():
    '''
    Cancels all bitcoin orders.
    '''
    client = gdax.AuthenticatedClient(API_KEY, API_SECRET, API_PASS)
    r = client.cancel_all(product='LTC-USD')
    print(json.dumps(r))


if __name__ == '__main__':
    main()
