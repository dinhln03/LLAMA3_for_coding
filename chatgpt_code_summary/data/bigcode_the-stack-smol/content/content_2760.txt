import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

def email_mapper(df):
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    user_item = df.groupby('user_id')['article_id'].value_counts().unstack()
    user_item[user_item.isna() == False] = 1
    
    return user_item # return the user_item matrix 


def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    article_id_grouped_df = df.groupby(['title'])
    top_articles = article_id_grouped_df['user_id'].count().sort_values(ascending=False).iloc[:n].index.tolist()
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    article_id_grouped_df = df.groupby(['article_id'])
    top_articles_ids = article_id_grouped_df['user_id'].count().sort_values(ascending=False).iloc[:n].index.tolist()

    return top_articles_ids # Return the top article ids



def user_user_recs(user_id, user_item, df, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    def get_user_articles_names_ids(user_id):
        '''
        INPUT:
        user_id


        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids 
                        (this is identified by the doc_full_name column in df_content)
        
        Description:
        Provides a list of the article_ids and article titles that have been seen by a user
        '''
        # Your code here
        article_ids = user_item.loc[user_id][user_item.loc[user_id] ==1].index.tolist()
        article_names = []
        for i in article_ids:
            try:
                title = df[df['article_id'] == i]['title'].unique()[0]
            except IndexError:
                title ="None"
                
            article_names.append(title)
        article_ids = list(map(str, article_ids))
    
        return article_ids, article_names # return the ids and names

    def find_similar_users():
        '''        
        OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first
        
        Description:
        Computes the similarity of every pair of users based on the dot product
        Returns an ordered
        
        '''
        # compute similarity of each user to the provided user
        user_item_tmp = user_item.copy()
        user_item_tmp[user_item_tmp.isna() == True] = 0 # 1. Make Nan to 0
        row = user_item_tmp.loc[user_id] # 2. Select a row
        result_dot = row@user_item_tmp.T # 3. Dot product of each of row of the matrix 
        result_dot.drop(labels = [user_id], inplace=True) # remove the own user's id
        most_similar_users = result_dot.sort_values(ascending=False).index.tolist()  # sort by similarity # create list of just the ids
        
        return most_similar_users # return a list of the users in order from most to least similar

    def get_top_sorted_users(most_similar_users):
        
        '''
        INPUT:
        most_similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first 
                
        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of articles viewed by the user - if a u
                        
        Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                        highest of each is higher in the dataframe
        
        '''
        # Make neighbor_id column
        df_user_id_grouped =df.groupby("user_id")
        df_user_id_grouped['article_id'].count().sort_values(ascending=False)
        neighbors_df = pd.DataFrame()
        neighbors_df['neighbor_id'] = most_similar_users
        
        # make similarity column
        user_item_tmp = user_item.copy()
        user_item_tmp[user_item_tmp.isna() == True] = 0 # 1. Make Nan to 0
        row = user_item_tmp.loc[user_id] # Select a row
        result_dot = row@user_item_tmp.T # Dot product of each of row of the matrix 
        result_dot.drop(labels = [user_id], inplace=True) # remove the own user's id
        similarity = result_dot.sort_values(ascending=False).values.tolist()[0:10] 
        neighbors_df['similarity'] = similarity
        
        # Make num_interactions column
        num_interactions = []
        for i in neighbors_df['neighbor_id']:
            counted_interaction = df_user_id_grouped['article_id'].count().loc[i]
            num_interactions.append(counted_interaction)
        neighbors_df['num_interactions'] = num_interactions
        neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)
        
        return neighbors_df # Return the dataframe specified in the doc_string
    
    recs = []
    rec_names =[]
    counter = 0
    # Get seen article ids and names from selected user id
    article_ids, article_names = get_user_articles_names_ids(user_id)
    # Make set to find unseen articles
    seen_ids_set = set(article_ids)
    most_similar_users = find_similar_users()[0:10]
    neighbors_df = get_top_sorted_users(most_similar_users)
    # Find similar users of the selected user
    similar_users_list = neighbors_df['neighbor_id'] # Get neighbor_df


    # Make recommendation list
    for sim_user in similar_users_list:
         if counter < m: 
            # Get seen article ids and names from similar users
            sim_article_ids, sim_article_names = get_user_articles_names_ids(sim_user)
            # Make dict (key: article_ids, value:article_names)
            sim_user_dict = dict(zip(sim_article_ids, sim_article_names)) 
            # Make set to find unseen articles
            sim_seen_ids_set = set(sim_article_ids)
            # Create set of unseen articles_ids
            unseen_ids_set = sim_seen_ids_set.difference(seen_ids_set)

            for i in unseen_ids_set: 
                if counter < m: 
                    recs.append(i)
                    rec_names.append(sim_user_dict[i])
                    counter += 1
    
    
    return recs, rec_names


###


def make_Tfidf_array(df_content):
    def tokenize(text):
        '''
        Function splits text into separate words and gets a word lowercased and removes whitespaces at the ends of a word. 
        The funtions also cleans irrelevant stopwords.
        Input:
        1. text: text message
        Output:
        1. Clean_tokens : list of tokenized clean words
        '''
        # Get rid of other sepcial characters   
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # Tokenize
        tokens = word_tokenize(text)
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
            clean_tokens.append(clean_tok)

        # Remove stop words    
        stopwords = nltk.corpus.stopwords.words('english')
        clean_tokens = [token for token in clean_tokens if token not in stopwords]

        return clean_tokens

    corpus = df_content['doc_description']
    df_content['doc_description'].fillna(df_content['doc_full_name'], inplace=True)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # Text Processing, Feature Extraction
    vect = TfidfVectorizer(tokenizer=tokenize)
    # get counts of each token (word) in text data
    X = vect.fit_transform(corpus)
    X = X.toarray()

    return vect, X


def make_content_recs(article_id, df_content, df, m=10):
    '''
    INPUT:
    article_id = (int) a article id in df_content
    m - (int) the number of recommendations you want for the user
    df_content -  (pandas dataframe) df_content as defined at the top of the notebook 
    df - (pandas dataframe) df as defined at the top of the notebook 

    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    '''
    def tokenize(text):
        '''
        Function splits text into separate words and gets a word lowercased and removes whitespaces at the ends of a word. 
        The funtions also cleans irrelevant stopwords.
        Input:
        1. text: text message
        Output:
        1. Clean_tokens : list of tokenized clean words
        '''
        # Get rid of other sepcial characters   
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # Tokenize
        tokens = word_tokenize(text)
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
            clean_tokens.append(clean_tok)

        # Remove stop words    
        stopwords = nltk.corpus.stopwords.words('english')
        clean_tokens = [token for token in clean_tokens if token not in stopwords]

        return clean_tokens
        
    vect, X = make_Tfidf_array(df_content)
    
    if article_id in df_content.article_id:
        cosine_similarity = linear_kernel(X, X)
        df_similarity = pd.DataFrame(cosine_similarity[article_id], columns=['similarity'])
        df_similarity_modified = df_similarity.drop(article_id)
        recs = df_similarity_modified.similarity.sort_values(ascending=False).index[0:10].tolist()
        rec_names = []

        for i in recs:
            name = df_content[df_content['article_id'] == i]['doc_full_name'].values[0]
            rec_names.append(name)

    else:
        tfidf_feature_name = vect.get_feature_names()
        # Get title of the document of interest
        booktitle = df[df['article_id'] == article_id]['title'].values[0]
        # Tokenize the title
        booktitle_tokenized = tokenize(booktitle)

        X_slice_list = []
        for i in booktitle_tokenized:
            if i in tfidf_feature_name:
                X_slice_list.append(tfidf_feature_name.index(i))

        X_slice_list.sort()
        X_sliced = X[:,X_slice_list]
        check_df = pd.DataFrame(X_sliced, columns=X_slice_list)
        check_df['sum'] = check_df.sum(axis=1)
        recs = check_df.sort_values("sum", ascending=False)[0:10].index.tolist()
        rec_names = []
        for i in recs:
            name = df_content[df_content['article_id'] == i]['doc_full_name'].values[0]
            rec_names.append(name)

    return recs, rec_names

