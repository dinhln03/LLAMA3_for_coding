import urllib.request,json
from .models import Sources, Articles
from datetime import datetime

#Getting api key
api_key = None
#Getting the news base url
# NEWS_API_KEY = None
# NEWS_API_BASE_URL = None
ARTICLE = None

def configure_request(app):
    global api_key,NEWS_API_BASE_URL,NEWS_API_KEY,ARTICLE
    api_key = app.config['NEWS_API_KEY']
    ARTICLE = app.config['ARTICLE']
    NEWS_API_BASE_URL = app.config['NEWS_API_BASE_URL']
    NEWS_API_KEY = app.config['NEWS_API_KEY']
    
def get_source(category):
    '''
    function that gets the json response to our url request
    '''
    get_source_url = NEWS_API_BASE_URL.format(category,api_key)
    print(get_source_url)
    
    with urllib.request.urlopen(get_source_url) as url:
        get_source_data = url.read()
        get_source_response = json.loads(get_source_data)
        
        sources_result = None
        
        if get_source_response['sources']:
            sources_results_list = get_source_response['sources']
            sources_result = process_sources(sources_results_list)
            print(sources_result)
            
            
    return sources_result  

def process_sources(sources_list):
    '''
    Function that checks the news results and turn them into objects
    
    Args:
        sources_list: A list of dictionaries that contain sources details
    '''
    sources_result = []
    
    for source_item in sources_list:
        author = source_item.get('author')
        title = source_item.get('title')
        imageurl = source_item.get('urltoimage')
        description = source_item.get('description')
        url = source_item.get('url')
        id = source_item.get('id')
        
        sources_object = Sources(author, title,imageurl,description,url,id)
        sources_result.append(sources_object)
        
    return sources_result

def get_articles(id):
    '''
    Function that processes the articles and returns a list of articles objects
    '''
    get_articles_url = ARTICLE.format(id,api_key)
    print(get_articles_url)
    
    with urllib.request.urlopen(get_articles_url) as url:
        article_data = url.read()
        articles_response = json.loads(article_data)
        
        articles_object = None
        if articles_response['articles']:
            response_list= articles_response['articles']
            articles_object = process_articles(response_list)
            
    return articles_object

def process_articles(articles_list):
    '''
    function that checks the articles and processes them into instances
    '''
    articles_object = []
    for article_item in articles_list:
        author = article_item.get('name')
        title = article_item.get('title')
        description = article_item.get('description')
        url = article_item.get('url')
        image = article_item.get('urlToImage')
        date = article_item.get('publishedAt')
        
        if image:
            articles_result = Articles(author,title,description,url,image,date)
            articles_object.append(articles_result)
            
            
    return articles_object        
        
    
    

