import praw
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent='testscript',
                     username=USERNAME,
                     password=PASSWORD)

subreddit = reddit.subreddit('python')
hot_python = subreddit.hot(limit=5)

for post in hot_python:
  if not post.stickied:
    print('Title: {}, Upvotes: {}, Downvotes: {}\n'.format(
                                                    post.title,
                                                    post.ups,
                                                    post.downs))