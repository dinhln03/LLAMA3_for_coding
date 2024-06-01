from bs4 import BeautifulSoup
from urllib.request import urlopen

def main():
    url = "http://www.networksciencelab.com"
    with urlopen(url) as doc:
        soup = BeautifulSoup(doc)

    links = [(link.string, link['href'])
        for link in soup.find_all('a')
        if link.has_attr('href')
    ]
    # print(links)

    broken_links = []
    for _, link in links:
        try:
            urlopen(link, timeout=3)
        except Exception as e:
            print(f'broken link {link}, {e}')
            broken_links.append(link)
    print(broken_links)

if __name__ == "__main__":
    main()