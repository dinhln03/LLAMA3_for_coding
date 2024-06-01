import requests


def gen_from_urls(urls: tuple) -> tuple:
    for resp in (requests.get(url) for url in urls):
        # yield returns only 1 items at a time.
        yield len(resp.content), resp.status_code, resp.url


if __name__ == "__main__":
    urls = (
        "https://www.oreilly.com/",
        "https://twitter.com/",
        "https://www.google.com/",
    )

    for resp_len, status, url in gen_from_urls(urls):
        print(resp_len, status, url)
