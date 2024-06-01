import pandas as pd
import numpy as np
import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from IPython.core.display import display, HTML

def art_search(art):
    '''
    Function to retrieve the information about collections in the Art institute of Chicago

    Parameters:
    -------------
    The key word that users want to search,
    for example: the artist's name, the title of the artwork.

    Returns:
    -------------
    Status code: str
        if the API request went through
    Dataframe: df
        includes the related info about the searched artworks.

    Example:
    -------------
    >>>art_search('monet')
       0	16568	Water Lilies	Claude Monet\nFrench, 1840-1926	France	1906	1906	Oil on canvas	[Painting and Sculpture of Europe, Essentials]
       1	16571	Arrival of the Normandy Train, Gare Saint-Lazare	Claude Monet\nFrench, 1840-1926	France	1877	1877	Oil on canvas	[Painting and Sculpture of Europe]
    '''
    params_search = {'q': art} 
    r = requests.get("https://api.artic.edu/api/v1/artworks/search?fields=id,title,date_start,date_end,artist_display,place_of_origin,medium_display,category_titles", params = params_search)        
    
    try:
        status = r.status_code
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('no error (successfully made request)')
        r1 = json.dumps(r.json(), indent = 2)
        artsearch = json.loads(r1)
        artworks = pd.DataFrame(artsearch['data'])
        artworks_info = artworks[['id','title','artist_display','place_of_origin','date_start','date_end','medium_display','category_titles']]
        
        return artworks_info
    
def tour_search(tour):
    '''
    Function to retrieve the information about tour in the Art institute of Chicago

    Parameters:
    -------------
    The key word that users want to search,
    for example: the artist's name, the title of the artwork.

    Returns:
    -------------
    Status code: str
        if the API request went through
    Dataframe: df
        includes the related info about the searched tour.

    Example:
    -------------
    >>>tour_search('monet')
       0	4714	Monet and Chicago	http://aic-mobile-tours.artic.edu/sites/defaul...	<p>Monet and Chicago presents the city’s uniqu...	<p>Monet and Chicago is the first exhibition t...	[Cliff Walk at Pourville, Caricature of a Man ...	[Claude Monet, Claude Monet, Claude Monet, Cla...
       1	4520	Manet and Modern Beauty	http://aic-mobile-tours.artic.edu/sites/defaul...	<p>Dive deep into the life and mind of one the...	<p>Manet is undoubtedly one of the most fascin...	[]	[]

    '''
    params_search_tour = {'q': tour} 
    rt = requests.get("https://api.artic.edu/api/v1/tours/search?fields=id,image,title,description,intro,artwork_titles,artist_titles", params = params_search_tour)
    try:
        status = rt.status_code
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('no error (successfully made request)')
        rt1 = json.dumps(rt.json(), indent = 2)
        toursearch = json.loads(rt1)
        ntour = pd.DataFrame(toursearch['data'])
        tour_info = ntour[['id','title','image','description','intro','artwork_titles','artist_titles']]
        
        return tour_info
 
def pic_search(pic, artist):
    '''
    Function to retrieve the images of artworks collected in the Art institute of Chicago

    Parameters:
    -------------
    pic: the title of the artwork
    artist: the full name of the artist

    Returns:
    -------------
    Status code: str
        if the API request went through
    Image: jpg
        The image of the searched atwork
    Error Message:
        Error messsage if the search is invalid

    Example:
    -------------
    >>>pic_search('Water Lillies', 'Claude Monet')

    '''
    params_search_pic = {'q': pic} 
    rp = requests.get("https://api.artic.edu/api/v1/artworks/search?fields=id,title,artist_display,image_id", params = params_search_pic)
    
    linkhead = 'https://www.artic.edu/iiif/2/'
    linktail = '/full/843,/0/default.jpg'
    
    try:
        status = rp.status_code
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('no error (successfully made request)')
        rp1 = json.dumps(rp.json(), indent = 2)
        picsearch = json.loads(rp1)
        npic = pd.DataFrame(picsearch['data'])
        pic_info = npic[['id','title','artist_display','image_id']]     
        
        df_len = len(pic_info)
        for i in range(df_len):
            if pic_info.iloc[i]['title'] == pic and (artist in pic_info.iloc[i]['artist_display']): # match title and artist with user input
                get_image_id = pic_info.iloc[i]['image_id']
                image_link = linkhead + get_image_id + linktail
                response = requests.get(image_link)
                img = Image.open(BytesIO(response.content))
                return img
                
        print("Invalid Search! Please find related information below :)")
        return pic_info
    
def product_search(product_art, product_category):
    '''
    Function to retrieve the information about products sold in the Art institute of Chicago

    Parameters:
    -------------
    pic: the title of the artwork
    artist: the full name of the artist

    Returns:
    -------------
    Status code: str
        if the API request went through
    DataFrame: a dataframe include related info about the products and images of the products

    Example:
    -------------
    >>>product_search('Rainy Day', 'Mug')
    >>>0	245410	Gustave Caillebotte Paris Street; Rainy Day Mug		$9.95...

    '''
    params_search_product = {'q': product_art} 
    rpro = requests.get("https://api.artic.edu/api/v1/products?search", params = params_search_product)

    try:
        status = rpro.status_code
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('no error (successfully made request)')
        rpro1 = json.dumps(rpro.json(), indent = 2)
        productsearch = json.loads(rpro1)
        nproduct = pd.DataFrame(productsearch['data'])
        df_len1 = len(nproduct)
        for i in range(df_len1):
            if product_art in nproduct.iloc[i]['title'] and (product_category in nproduct.iloc[i]['description']): # match title and artist with user input
                product_info = nproduct[['id','title','image_url','price_display','description']]
                
                def path_to_image_html(path):
                    return '<img src="'+ path + '" width="60" >'
                image_cols = ['image_url']

                format_dict={}
                for image_cols in image_cols:
                    format_dict[image_cols] = path_to_image_html
                html = display(HTML(product_info.to_html(escape = False,formatters = format_dict)))
                return html
            else:
                return"Invalid Search! Please try other artworks or categories:)"
    
def product_show(product_art_show):
    '''
    Function to retrieve the information about top10 products sold in the Art institute of Chicago

    Parameters:
    -------------
    Type in any random word

    Returns:
    -------------
    Status code: str
        if the API request went through
    DataFrame: a dataframe include related info about the top 10 products and images of the products

    Example:
    -------------
    >>>product_search('')
    >>>0	250620	The Age of French Impressionism—Softcover		$30...

    '''
    params_show_product = {'q': product_art_show} 
    rproshow = requests.get("https://api.artic.edu/api/v1/products?limit=10", params = params_show_product)

    try:
        status = rproshow.status_code
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('no error (successfully made request)')
        rproshow1 = json.dumps(rproshow.json(), indent = 2)
        productshow = json.loads(rproshow1)
        nproductshow = pd.DataFrame(productshow['data'])
        product_show_info = nproductshow[['id','title','image_url','price_display','description']]
                
        def path_to_image_html(path):
            return '<img src="'+ path + '" width="60" >'
        image_cols1 = ['image_url']

        format_dict={}
        for image_cols1 in image_cols1:
            format_dict[image_cols1] = path_to_image_html
        html1 = display(HTML(product_show_info.to_html(escape = False,formatters = format_dict)))
        return html1