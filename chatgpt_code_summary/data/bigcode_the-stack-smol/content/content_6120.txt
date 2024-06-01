from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
import os
import argparse


def shorten_link(token, url):
    response = requests.post(
        "https://api-ssl.bitly.com/v4/bitlinks",
        headers={"Authorization": "Bearer {}".format(token)},
        json={"long_url": url})
    response.raise_for_status()
    return response.json()["link"]


def count_clicks(token, link):
    response = requests.get(
        "https://api-ssl.bitly.com/v4/bitlinks/{0}{1}/clicks/summary"
            .format(link.netloc, link.path),
        headers={"Authorization": "Bearer {}".format(token)})
    response.raise_for_status()
    return response.json()["total_clicks"]


def is_bitlink(token, link):
    response = requests.get(
        "https://api-ssl.bitly.com/v4/bitlinks/{0}{1}"
            .format(link.netloc, link.path),
        headers={"Authorization": "Bearer {}".format(token)})
    return response.ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Программа для сокращения ссылок или "
                    "подсчёта количества переходов для bitlink")
    parser.add_argument("url", help="Введите URL или bitlink")
    args = parser.parse_args()
    link = args.url
    parsed_bitlink = urlparse(link)
    load_dotenv()
    token = os.environ["BITLY_TOKEN"]

    try:
        if is_bitlink(token, parsed_bitlink):
            clicks_count = count_clicks(token, parsed_bitlink)
            print("Количество переходов по вашей ссылке: ", clicks_count)
        else:
            bitlink = shorten_link(token, link)
            print("Сокращенная ссылка: ", bitlink)
    except:
        print("Вы ввели неправильную ссылку")
