import config
from twitter import *


def main():
    t = Twitter(
        auth=OAuth(config.TW_TOKEN, config.TW_TOKEN_SECRET, config.TW_CONSUMER_KEY, config.TW_CONSUMER_SECRET))
        
    # Post a message
    msg = 'テスト投稿ですm(_ _)m'
    t.statuses.update(status=msg)


if __name__ == '__main__':
    main()
