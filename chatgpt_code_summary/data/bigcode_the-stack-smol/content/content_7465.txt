import re
import os
import time
import tweepy
import neologdn
import emoji

TRAINFILE = "./data/train.tsv"
DEVFILE = "./data/dev.tsv"
TESTFILE = "./data/test.tsv"

dup = []
if os.path.exists(TRAINFILE):
    with open(TRAINFILE) as f:
        dup += f.readlines()
if os.path.exists(DEVFILE):
    with open(DEVFILE) as f:
        dup += f.readlines()
if os.path.exists(TESTFILE):
    with open(TESTFILE) as f:
        dup += f.readlines()


class Tweet:
    def __init__(self, status):
        self.in_reply_to_status_id = status.in_reply_to_status_id
        self.text = status.text
        self.created_at = status.created_at
        self.screen_name = status.user.screen_name
        self.username = status.user.name
        self.user_id = status.user.id


def is_valid_tweet(status):
    # is bot
    if "bot" in status.user.screen_name:
        return False
    # include URL
    if re.search(r"https?://", status.text):
        return False
    # is hashtag
    if re.search(r"#(\w+)", status.text):
        return False
    # reply to multi user
    tweet = re.sub(r"@([A-Za-z0-9_]+)", "<unk>", status.text)
    if tweet.split().count("<unk>") > 1:
        return False
    # too long
    if len(tweet.replace("<unk>", "")) > 20:
        return False
    return True


def normalize(text):
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = neologdn.normalize(text)
    text = "".join(["" if c in emoji.UNICODE_EMOJI["en"].keys() else c for c in text])
    tmp = re.sub(r"(\d)([,.])(\d+)", r"\1\3", text)
    # text = re.sub(r"\d+", "0", tmp)
    tmp = re.sub(r"[!-/:-@[-`{-~]", r" ", text)
    text = re.sub(u"[■-♯]", " ", tmp)
    text = text.strip()
    return text


def main():
    query = input("Search Query: ")
    max_tw = int(input("Tweet Count: "))
    CK = os.getenv("TW_CK")
    CS = os.getenv("TW_CS")
    auth = tweepy.AppAuthHandler(CK, CS)
    api = tweepy.API(auth)
    got = 0
    filtered = 0
    saved = 0
    lookup_ids = []
    replies = {}
    max_id = None
    while saved <= max_tw:
        try:
            statuses = api.search_tweets(
                q=query, lang="ja", count=100, max_id=max_id
            )
            max_id = statuses[-1].id
            for status in statuses:
                got += 1
                # is not reply
                if not status.in_reply_to_status_id:
                    continue
                # filter
                if not is_valid_tweet(status):
                    continue
                # append lookup id
                lookup_ids.append(status.in_reply_to_status_id)
                replies[status.in_reply_to_status_id] = Tweet(status)
                filtered += 1
                print(f"\r{got} => {filtered} => {saved}", end="")
                # collect 100 tweets
                if len(lookup_ids) >= 100:
                    pstatuses = api.lookup_statuses(lookup_ids)
                    for pstatus in pstatuses:
                        if not is_valid_tweet(pstatus):
                            continue
                        reply = replies[pstatus.id]
                        # is same user
                        if pstatus.user.id == reply.user_id:
                            continue
                        intext = re.sub(r"@([A-Za-z0-9_]+)", "", pstatus.text)
                        intext = normalize(intext)
                        outtext = re.sub(r"@([A-Za-z0-9_]+)", "", reply.text)
                        outtext = normalize(outtext)
                        if not intext or not outtext:
                            continue
                        if f"{intext}\t{outtext}\n" in dup:
                            continue
                        if saved <= max_tw * .9:
                            path = TRAINFILE
                        elif saved <= max_tw * .95:
                            path = DEVFILE
                        else:
                            path = TESTFILE
                        with open(path, "a") as f:
                            f.write(f"{intext}\t{outtext}\n")
                        saved += 1
                        print(f"\r{got} => {filtered} => {saved}", end="")
                        if saved > max_tw:
                            exit()
                    lookup_ids = []
                    replies = {}
        except Exception:
            print()
            limit_status = api.rate_limit_status(
            )["resources"]["search"]["/search/tweets"]
            while limit_status["reset"] >= int(time.time()):
                print("\rLimited: " + ("   " +
                      str(limit_status["reset"] - int(time.time())))[-3:] + "s", end="")
                time.sleep(.5)
            print()


if __name__ == "__main__":
    main()
