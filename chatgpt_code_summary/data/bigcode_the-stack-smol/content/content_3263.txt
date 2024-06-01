import praw

c_id='34kxuaxc4yWiKw'
c_secret='8bJqHqNHFdB6NKV9sHzFbo4_Dl4'
ua='my user agent'
un='the_ugly_bot'
pwd='whatever930'

def  login():
	r = praw.Reddit(client_id=c_id,
                     client_secret=c_secret,
                     user_agent=ua,
                     username=un,
                     password=pwd)
	return r