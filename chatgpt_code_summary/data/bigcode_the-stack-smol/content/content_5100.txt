import aiohttp, asyncio
from bs4 import BeautifulSoup
import json
import time

VC_SEARCH = "https://vc.ru/search/v2/content/new"


async def parse_urls(key_word):
    async with aiohttp.ClientSession() as session:
        async with session.get(VC_SEARCH, params={
            "query": key_word,
            "target_type": 'posts',
        }) as r:
            soup = BeautifulSoup(await r.text(), 'html.parser')
            urls = [x["href"] for x in soup.find_all("a", {"class": "content-feed__link"})]

            return urls


async def get_text(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            soup = BeautifulSoup(await r.text(), 'html.parser')
            text = " ".join(map(lambda x: x.text, soup.find("div", {"class": "l-entry__content"}).find_all("p")))
            return text


async def get_all_texts(keyword):
    urls = await parse_urls(keyword)
    all_texts = []
    for u in urls[:25]:
        text = await get_text(u)
        all_texts.append(text)
    return all_texts


async def vc_get_data(keyword, result_file_path='result-vc.json'):
    texts = await get_all_texts(keyword)
    result_dict = {"company": keyword,
                   "texts": texts}
    result_json = json.loads(json.dumps(result_dict))
    return result_json
    #with open(result_file_path, 'w', encoding='utf-8') as f:
       # json.dump(result_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(vc_get_data("сбер", "other/sber-vc.json"))
