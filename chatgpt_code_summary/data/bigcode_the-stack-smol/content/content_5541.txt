#!/usr/bin/python
# -*- coding: utf-8 -*-

import re

import core.pywikibot as pywikibot

from utils import (
    SITE_NAMES,
    create_links_string,
    read_log,
    generate_list_page,
)


def main():
    site = pywikibot.Site()
    for site_name in SITE_NAMES:
        page_name, site_regexp, list_page = generate_list_page(site, site_name)

        list_page.text = list_page.text + '\n'
        bad_pages_count = int(re.findall(r'Текущее количество: (\d+)', list_page.text)[0])
        read_pages_count = 0

        for string in list_page.text.split('\n'):
            if not string or string[0] != '#':
                continue
            title = re.findall(r'\[\[(.+?)]]', string)[0]
            page = pywikibot.Page(site, f'{title}')
            links = [link for link in re.findall(site_regexp, page.text, flags=re.I)]

            if not links:
                list_page.text = list_page.text.replace(f'{string}\n', '')
                bad_pages_count -= 1
            else:
                links_string = create_links_string(links, page)
                list_page.text = list_page.text.replace(string, links_string[:-1:])
            read_pages_count += 1
            read_log(read_pages_count)

        list_page.text = re.sub(r'Текущее количество: (\d+)', fr'Текущее количество: {bad_pages_count}', list_page.text)
        list_page.save(u'обновление ссылок')


if __name__ == '__main__':
    main()
