from gzip import decompress
from http import cookiejar
from json import loads, dumps
from os import environ
from time import strftime, gmtime
from urllib import request


def get_url(ticker):
    env = environ.get('FLASK_ENV', 'development')

    if env == 'development':
        url = 'https://www.fundamentus.com.br/amline/cot_hist.php?papel='
    else:
        phproxy = 'http://shortbushash.com/proxy.php'
        url = phproxy + '?q=https%3A%2F%2Fwww.fundamentus.com.br%2Famline%2Fcot_hist.php%3Fpapel%3D'

    return url + ticker + '&hl=1a7', env


def build_headers(url, env):
    if env == 'development':
        headers = [
            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'),
            ('Referer', url),
            ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'),
        ]
    else:
        headers = [
            ('Accept', 'application/json, text/javascript, */*; q=0.01'),
            ('Accept-Encoding', 'gzip, deflate, br'),
            ('Referer', url),
            ('User-Agent', 'PostmanRuntime/7.26.8'),
        ]

    return headers


def parse_epoch_time(parsed_content):
    return [[strftime('%Y-%m-%d', gmtime(unix_epoch_time/1000)), price] for [unix_epoch_time, price] in parsed_content]


def load_prices(ticker, parse_json=True):
    url, env = get_url(ticker)
    cookie_jar = cookiejar.CookieJar()
    opener = request.build_opener(request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = build_headers(url, env)

    with opener.open(url) as link:
        gzip_response = link.read()
        binary_response = gzip_response.decode() if env == 'development' else decompress(gzip_response)
        parsed_content = loads(binary_response)
        content = parse_epoch_time(parsed_content)

    return dumps(content) if parse_json else content
