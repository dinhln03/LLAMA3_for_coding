import os
import re


def gen_sitemap(main_site, md_file):
    pattern = re.compile(r': (.*?).md', re.S)
    res = []
    with open(md_file) as md:
        for line in md.readlines():
            line = str(line)
            cur_urls = re.findall(pattern, line)
            if len(cur_urls) > 0:
                if cur_urls[0] == '/':
                    continue
                res.append(main_site + cur_urls[0])

    return res


if __name__ == '__main__':
    print("生成wiki站的sitemap")
    site_map = gen_sitemap('https://www.an.rustfisher.com/',
                           '/Users/rustfisher/Desktop/ws/wiki-ws/mk-android-wiki-proj/mk-an-wiki/mkdocs.yml')
    print(len(site_map))

    sitemap_file = 'a-sp.txt'
    if os.path.exists(sitemap_file):
        os.remove(sitemap_file)
    with open(sitemap_file, 'w') as s:
        for url in site_map:
            s.write(url)
            s.write('\n')
