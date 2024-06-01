'''
Created on 2012/09/03

@author: amake
'''

from __future__ import print_function
import os
import sys
import urllib
import codecs
from datetime import datetime
from xml.etree import ElementTree
import putio

CACHE_FILE = "cache.txt"
FEEDS_FILE = "feeds.txt"

DEBUG = True

PUTIOAPI = None


# Stupid CloudFlare decided to block "non-standard" browsers.
# Spoofing the user-agent gets around it.
class CustomURLopener(urllib.FancyURLopener):
    version = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) '
    'AppleWebKit/536.26.17 (KHTML like Gecko) Version/6.0.2 Safari/536.26.17'


urllib._urlopener = CustomURLopener()


def log(message):
    if DEBUG:
        print(message.encode('utf-8'))


class feedputter():
    '''
    Grab torrent files from an RSS feed.
    '''

    def __init__(self, feed):
        '''
        Constructor
        '''
        self.feed = feed
        self.cache = []
        if os.path.isfile(CACHE_FILE):
            self.cache = [line.strip() for line in codecs.open(
                CACHE_FILE, 'r', 'utf-8').readlines()]

    def __get_items(self):
        log("Fetching feed from: %s" % self.feed)

        data = urllib.urlopen(self.feed).read()
        tree = ElementTree.fromstring(data)

        return tree.findall(".//item")

    def save_torrent(self, link, target, title):

        torrent = urllib.urlopen(link)

        if (torrent.getcode() != 200):
            log("Error " + torrent.getcode())
            return False

        with open(os.path.join(target, title + ".torrent"), "w") as out:
            out.write(torrent.read())

        return True

    def putio(self, link, target, title):

        api = putio.get_api(target_folder=target)

        try:
            api.add(link, putio.CALLBACK_URL + '?amk_type=tv')

        except Exception as e:
            print(e)
            print('Skipping.')
            return False

        return True

    def get_to(self, target, method):
        '''
        Fetch linked torrents and save to the specified output folder.
        '''

        for item in self.__get_items():

            title = item.find('title').text.strip()
            link = item.find('link').text

            log("Found " + title)

            if title in self.cache:
                log("Already gotten. Skipping.")
                continue

            log("Getting ... ")

            if not method(link, target, title):
                continue
            with codecs.open(CACHE_FILE, "a", "utf-8") as tmp:
                tmp.write(title + "\n")

            log("Done")


def usage():
    print('Usage: {0} TARGET_DIR'.format(os.path.basename(__file__)))


def main():

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    if not os.path.isdir(sys.argv[1]):
        print('Directory not found or not a directory:', sys.argv[1])
        print()
        usage()
        sys.exit(1)

    os.chdir(os.path.dirname(__file__))

    feeds = [line.strip() for line in open(FEEDS_FILE).readlines()]

    log(datetime.now().isoformat(" ") +
        " Starting feedputter with {0} feeds".format(len(feeds)))

    for feed in feeds:
        getter = feedputter(feed)
        getter.get_to(sys.argv[1], getter.putio)

    log(datetime.now().isoformat(" ") + " Finished feedputter")


if __name__ == "__main__":
    main()
