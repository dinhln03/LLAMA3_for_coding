'''
First Version: Tweets contents from subreddits
'''
from time import sleep
from reddit import Reddit
from twitter import Twitter


def main():
    '''
    connects the pieces to grab posts from reddit and throw them on twitter
    '''
    reddit = Reddit()
    twitter = Twitter()
    tweets = reddit.get_tweets()
    print("sending {} tweets".format(len(tweets)))
    for tweet in reddit.get_tweets():
        status = twitter.send_tweet(tweet.Primary)
        if tweet.Second:
            twitter.send_tweet(tweet.Second, status.id)
        sleep(90)


if __name__ == '__main__':
    main()
