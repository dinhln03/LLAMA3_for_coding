from bs4 import BeautifulSoup

import article
import requester

root_url = "http://www.nec-nijmegen.nl/"
source_url = "https://www.nec-nijmegen.nl/nieuws.htm"


def make_request():
    html = requester.get_html(source_url)
    return html


def get_articles():
    print("Getting articles from: " + source_url)
    html = make_request()
    soup = BeautifulSoup(html, "html.parser")
    lis = soup.find_all("div", class_="item")

    arts = []

    for eles in lis:
        print()
        art = article.Article()
        art.title = eles.a.get("title")
        art.url = eles.a.get("href")
        art.full_url = root_url + art.url
        art.source = source_url
        arts.append(art)
    return arts
