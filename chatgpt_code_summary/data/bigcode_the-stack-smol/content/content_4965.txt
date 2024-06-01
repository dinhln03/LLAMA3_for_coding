import tweepy
from time import sleep
from datetime import datetime
from keys import *
from tqdm import tqdm

def play(test=True,i_pages=3, i_hashtag=20, like_pages=False, like_hashtag=False):
  
    while True == True:
        try:
            econotwbot(test, i_pages, i_hashtag)
            
        except Exception as e:
            print(e)
            sleep(60*30)
            pass
            

class econotwbot:
    
    def __init__(self, test=True, i_pages=3, i_hashtag=20, like_pages=False, like_hashtag=False):
        
        self.test = test
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
        self._file_following = 'to_follow.txt'
        self.i_pages = i_pages
        self.i_hashtag = i_hashtag
        self.like_pages = like_pages
        self.like_hashtag = like_hashtag
        self.pbar = 0
        self.iteractions = self.i_pages*len(self.following_pages())+self.i_hashtag
        
        if self.test==True:
            self.hellow_world()
            self.retweepy()
            
        while self.test==False:
            print()
            print("Starting!",datetime.now())
            print()
            with tqdm(total=self.iteractions) as self.pbar:
                self.retweepy_following()
                tqdm.write("Just give me 5 more minutes to sleep please!")
                sleep(5*60)
                self.retweepy()
            print()
            print("Iteraction done!",datetime.now())			
            sleep(30*60)
    
    def following_pages(self):
        with open(self._file_following, 'r') as f:
            return f.read().splitlines()
    
    def retweepy(self):
        tqdm.write('Delivering tweets with #econotw OR #EconTwitter')
        dici={'q':'#econotw OR #EconTwitter'}
        args={'method':self.api.search,
              'dici':dici,
              'like':self.like_hashtag,
              'i':self.i_hashtag}
        self.like_and_rt(**args)
    
    def retweepy_following(self):
        tqdm.write('Delivering interesting tweets')
        for page in self.following_pages():
            dici={'screen_name':page}
            args={'method':self.api.user_timeline,
                  'dici':dici,
                  'like':self.like_pages,
                  'i':self.i_pages}
            self.like_and_rt(**args)

    def like_and_rt(self,method,dici,like,i):
        count=0
        for tweet in tweepy.Cursor(method=method,**dici).items(i):
            self.pbar.update(1)
            count+=1
            try:
                if like==True:
                    self.api.create_favorite(tweet.id)
                    sleep(1)
                tweet.retweet()
                string= 'Retweeted: '+ str(tweet.id) +' @'+tweet.user.screen_name
                tqdm.write(string)
                sleep(10)
                
            # Print retweet errors
            except tweepy.TweepError as error:
                if (eval(error.reason)[0]['code'] != 139) and (eval(error.reason)[0]['code'] != 327):
                    tqdm.write('\nError. '+str(tweet.id)+' Retweet not successful. Reason: ')
                    tqdm.write(str(error.reason) +' '+ str(datetime.now()))
                    self.pbar.update(i-count)
        
            except StopIteration:
                break
 
            
    def hello_world(self):
        self.api.update_status("""Hello World! #econotw""")

if __name__ == "__main__":
    play(test=False)