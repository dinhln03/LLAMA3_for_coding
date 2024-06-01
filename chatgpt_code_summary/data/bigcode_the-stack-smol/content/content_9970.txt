# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:00:26 2018

@author: Alex

# reads and parses local html 

"""

#%% Import libraries
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import codecs
import os 
import re
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from gensim import models
import pickle
from sklearn.feature_extraction import stop_words


#%% Read in saved html
# read in saved html back in
def read_local_html(blog_folder,blog_num):
    
    # make filename
    filename = blog_folder + 'kd_blog' + str(blog_num).zfill(4) + '.html'
    
    # read in file    
    f = codecs.open(filename, 'r', 'utf-8')
    
    # parse file
    soup = BeautifulSoup(f.read(), 'html.parser')

    return soup

#%% 
def get_article_str(soup):
    
    # Titles 
    title = soup.title.text
    
    # Tag data
    tag = soup.find_all('div', class_ = 'tag-data')
    tags = tag[0].text
    tags = tags.replace('Tags: ','')
            
    # Paragraphs
    paras = soup.find_all('p')
    
    # The first paragraph always contains a description of the article
    description = paras[0].text
    
    # Get main text
    main_text = ""
    # remove second paragraph if it just contains author name
    if "By " not in paras[1].text:
        main_text = paras[1].text
        
    for i in range(2,len(paras)):
        # These if statements remove later paragraphs if they don't contain the main text of the article 
        if i > len(paras)-5 and "Bio" in paras[i].text:
            continue
        elif i > len(paras)-5 and "Original" in paras[i].text:
            continue
        elif i > len(paras)-5 and "Related" in paras[i].text:
            continue
        elif i > len(paras)-5 and "disqus" in paras[i].text:
            continue
        elif i > len(paras)-5 and "Pages" in paras[i].text:
            continue
        else:
            main_text = main_text + ' ' + paras[i].text
        
    # Create an article string 
    article_str = title + '. ' + tags + '. ' + description + ' ' + main_text
    
    return article_str

#%%
def clean_article(article_str):   
    # lowercase
    article_str = article_str.lower()

    #Remove any non alphanumeric characters that are no end-of-sentence punctuation
    article_str = re.sub('[^a-z\s\.\?\!]+','', article_str)
    
    # Replace ? with . 
    article_str = re.sub('\?','.', article_str)
    # Replace ! with . 
    article_str = re.sub('\!','.', article_str)
    
    # Replace more than one whitespace with one whitespace
    article_str = re.sub('\s+',' ', article_str)
    # Remove trailing whitespace 
    article_str = re.sub("\s+(?!\S)", "",article_str)
    # Remove preceding whitespace 
    article_str = re.sub("(?<!\S)\s+", "",article_str)
    
    # Replace funny words from lemmatization
    article_str = re.sub("datum","data",article_str)
    article_str = re.sub("learn\s","learning",article_str)
    article_str = re.sub("miss","missing",article_str)
    
    
    return article_str

#%% Split each blog post into sentences 
def get_sentences(article_str):
    # lowercase
    article_str = article_str.lower()

    #Remove any non alphanumeric characters
    article_str = re.sub('[^a-z\s\.]+','', article_str)
    article_str = re.sub('\s+',' ', article_str)
    
    # Split doc into sentences 
    sent_text = nltk.sent_tokenize(article_str)
    
    # Split sentences into words 
    tokenized_sentences = []
    for sentence in sent_text:
        # remove periods 
        sentence = re.sub('\.','', sentence)
        # tokenize
        tokenized_sentences.append(nltk.word_tokenize(sentence))
        
    return tokenized_sentences

#%% 
def lemmatize(cleaned_article):
    nlp = spacy.load('en', disable=['parser', 'ner'])

    doc = nlp(article_str)
    lemma_article = " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc])

    cleaned_lemma = clean_article(lemma_article)
    return cleaned_lemma

#%% Extract phrases from all the documents 
def phrase_extractor(doc_sents):
    '''
    doc_sents is a list where each element is a list with elements corresponding to individual sentences of a document
    '''
    # rename some functions
    Phraser = models.phrases.Phraser
    Phrases = models.phrases.Phrases
    
    # Generate list of sentences 
    sentence_stream = sum(doc_sents, [])
    
    # Generate bigrams
    common_terms = ["of", "with", "without", "and", "or", "the", "a", "as"]
    phrases = Phrases(sentence_stream, common_terms=common_terms)
    bigram = Phraser(phrases)
    
    # Generate trigrams 
    trigram = Phrases(bigram[sentence_stream])
    
    # Generate output
    output_strs = []
    for idx in range(0,len(doc_sents)):
        doc = doc_sents[idx]
        output_doc = list(trigram[doc])
        output_str = sum(output_doc,[])
        output_strs.append(' '.join(output_str))
        
    return output_strs

#%% Loop through all the blog posts 
blog_folder = 'C:\\Users\\Alex\\Documents\\GitHub\\insight-articles-project\\data\\raw\\kd_blogs\\'
os.chdir(blog_folder)
num_blog_posts = len(os.listdir(blog_folder))
documents = []
num_skipped = 0
blogs_included = []
doc_sents = []

for blog_num in range(1,num_blog_posts+1):
    try:
        # Parse html 
        soup = read_local_html(blog_folder,blog_num)
        article_str = get_article_str(soup)
        cleaned_article = clean_article(article_str)
        lemma_article = lemmatize(cleaned_article)
        
        # Extract sentences for phrase extraction 
        tokenized_sentences = get_sentences(lemma_article)
        doc_sents.append(tokenized_sentences)
        
        # Meta data 
        blogs_included.append(blog_num)
        
        
    except:
        print('Blog ' + str(blog_num) + ' skipped')
        num_skipped += 1


documents = phrase_extractor(doc_sents)

#documents.append(cleaned_article)     

   
# Save documents 
processed_data_folder = 'C:\\Users\\Alex\\Documents\\GitHub\\insight-articles-project\\data\\processed\\'
filename = processed_data_folder + 'kd_docs'

with open(filename, 'wb') as fp:
    pickle.dump((documents,blogs_included), fp)

'''
filename = processed_data_folder + 'doc_sents'

with open(filename, 'wb') as fp:
    pickle.dump(doc_sents, fp)
'''