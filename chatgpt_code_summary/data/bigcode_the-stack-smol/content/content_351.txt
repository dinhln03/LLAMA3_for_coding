from configparser import ConfigParser
import feedparser
import re
import requests
import tweepy


def get_id(xkcd_link: str) -> int:
    """
    Exctract comic id from xkcd link
    """
    match = re.search(r"\d+", xkcd_link)
    if match:
        return int(match.group())
    else:
        return 0


def get_xkcd_rss_entries(url: str):
    """
    Load latest XKCD RSS feed and extract latest entry
    """
    # get latest rss feed
    feed = feedparser.parse(url)

    return feed.get("entries")


def get_latest_rss_entry(entries: list):
    """
    Extract latest entry from XKCD RSS feed and
    parse the ID
    """
    entry = entries[0]
    id_ = get_id(xkcd_link=entry.get("id"))
    return id_, entry


def downdload_comic(entry: dict, filename: str) -> None:
    """
    Download latest image and store it in
    current working directory
    """
    match = re.search(r'src="(.*png)"', entry["summary"])
    if match:
        img_url = match.groups()[0]
        r = requests.get(img_url)
        r.raise_for_status()

        with open(filename, "wb") as f:
            f.write(r.content)
    return None


def initialize_twitter_api(config: ConfigParser):
    """
    Do authentication and return read-to-use
    twitter api object
    """
    twitter_config = config["twitter"]
    auth = tweepy.OAuthHandler(
        twitter_config.get("consumer_key"), twitter_config.get("consumer_secret")
    )
    auth.set_access_token(
        twitter_config.get("access_token"), twitter_config.get("access_secret")
    )

    api = tweepy.API(auth)

    return api


def send_twitter_post(entry: dict, api: tweepy.API, img_fname: str) -> None:
    """
    Post tweet on twitter
    """
    match = re.search("title=(.*)/>", entry["summary"])

    if match:
        msg = match.groups()[0]
        msg += f"\n {entry['link']}"
    else:
        msg = "-- No Title --"

    api.update_with_media(status=msg, filename=img_fname)

    return None
