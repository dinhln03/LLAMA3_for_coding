#!/usr/bin/env python
import argparse
import bs4
import os
import re


# better indentation hack via https://stackoverflow.com/a/15513483/127114
orig_prettify = bs4.BeautifulSoup.prettify
r = re.compile(r'^(\s*)', re.MULTILINE)
def prettify(self, encoding=None, formatter="minimal", indent_width=3):
    return r.sub(r'\1' * indent_width, orig_prettify(self, encoding, formatter))
bs4.BeautifulSoup.prettify = prettify


def process_file(filepath):
    """
    Rewrite links in `filepath` as follows: /some/path/index.html --> /some/path/
    """
    # print('processing', filepath)
    if filepath.endswith('.html'):

        # 1. read
        with open(filepath, 'r') as htmlfile:
            page = bs4.BeautifulSoup(htmlfile.read(), 'html.parser')

        # 2. rewrite links
        links = page.find_all('a')
        for link in links:
            href = link['href']
            if href.endswith('index.html'):
                href = href.replace('index.html', '')
                link['href'] = href


        # 3. hack to rewrite subtitle links that wget doesn't handle correctly
        video = page.find('video')
        if video:
            source = video.find('source')
            main_file = source['src']
            tracks = video.find_all('track')
            if tracks:
                for track in tracks:
                    # track_src = track['src']
                    # new_src = os.path.basename(track_src)
                    new_src = main_file.replace('.mp4', '.vtt')
                    track['src'] = new_src

        # 4. write
        with open(filepath, 'w') as htmlfile:
            html = page.prettify()
            htmlfile.write(html)


def deindexify(webroot):
    """
    Walks directory stucutre starting at `webroot` and rewrites all folder links.
    """
    content_folders = list(os.walk(webroot))
    for rel_path, _subfolders, filenames in content_folders:
        # print('processing folder ' + str(rel_path))
        for filename in filenames:
            filepath = os.path.join(rel_path, filename)
            if filepath.endswith('_Subtitle.vtt'):
                video_matching_filepath = filepath.replace('_Subtitle.vtt', '_Low_Resolution.vtt')
                os.rename(filepath, video_matching_filepath)
            else:
                process_file(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("webroot", help="Directory where website is stored.")
    args = parser.parse_args()
    deindexify(args.webroot)
    print('Removing index.html from folder links done.')

