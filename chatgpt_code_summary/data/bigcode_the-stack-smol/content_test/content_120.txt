import pandas as pd
import datetime
import matplotlib.pyplot as plt
import ast
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer 
import datetime



stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 



"""
Dates and dico
"""

df_sentiment = pd.read_excel('/Users/etiennelenaour/Desktop/Stage/vocab_sentiment.xlsx')

project_directory = '/Users/etiennelenaour/Desktop/Stage/'
l_month = ['January','February','March','April','May','June','July','August','September','October','November','December']
l_dates = list()


with open ('/Users/etiennelenaour/Desktop/Stage/csv_files/dates_fomc.csv', 'r') as doc :
    head = doc.readline()
    dates = doc.readlines()
    dates_to_chg = []
    for line in dates :
        if line.split(',')[1] == ' Y' :
            dates_to_chg += [line.split(';')[0]]
            date = 0
            m = 1   
            for month in l_month :
                if month[:3] == line.split(';')[0].split('/')[0] :
                    date += 100 * m
                m += 1
            date += int(line.split(',')[0].split('/')[2])*10000
            date += int(line.split(',')[0].split('/')[1])
            l_dates.append(date)

l_dates_final = l_dates[101:]

date_to_append = [20120125, 20120425, 20120620, 20120801, 20120913, 20121024, 20121212, 20130130,
20130130, 20130320, 20130501, 20130619, 20130918, 20131030, 20131218, 20140129,
20140129, 20140430, 20140618, 20140917, 20141029, 20141217]


for date in date_to_append:
    l_dates_final.append(date)

"""
cleaning functions
"""


def clean_dico_new_line(dico):

	new_dico = defaultdict(lambda: list())

	for keys, list_dico in dico.items():

		new_liste = [string.rstrip("\\n").lower() for string in list_dico]
		new_dico[keys] = new_liste


	return new_dico


def remove_stop_word(dico):


	new_dico = defaultdict(lambda: list())

	for keys, list_dico in dico.items():

		final_list = list()

		for ele in list_dico:

		    if (ele not in STOPWORDS) and (ele not in stop_words):
		        final_list.append(ele)

		new_dico[keys] = final_list

	return new_dico


def remove_nan_from_list(liste):

	new_liste = list()

	for ele in liste:
		if type(ele) == str:
			new_liste.append(ele)
		else:
			pass

	return new_liste


"""
Score functions
"""     

negative_word_list = [ele.lower() for ele in df_sentiment.Negative.tolist()]
positive_word_list = [ele.lower() for ele in remove_nan_from_list(df_sentiment.Positive.tolist())]

def compute_positivity(dico):
    """  This computes the positivity score of each statement. 
    Takes a dictionary with each statement as liste item and the corresponding interlocutor's name in names item  
    
    """ 
    dico_score = defaultdict(lambda: list())
    for name, liste in dico.items():
        neg_score = 0
        pos_score = 0
        for ele in liste:
            if ele in negative_word_list:
                neg_score += 1
            elif ele in positive_word_list:
                pos_score += 1
            else:
                pass
        if neg_score < 30 or pos_score < 30:
            pass
        else:
            score = (pos_score - neg_score) / (pos_score + neg_score)
            dico_score[name] = score
    return dico_score


def compute_mean_positivity(dico):

	neg_score = 0
	pos_score = 0

	for liste in dico.values():

		for ele in liste:

			if ele in negative_word_list:
				neg_score += 1

			elif ele in positive_word_list:
				pos_score += 1


			else:
				pass

	score = (pos_score - neg_score) / (pos_score + neg_score)

	return score

"""
Date function
"""


def from_int_dates(integ):
    string = str(integ)
    new_string = string[0]+ string[1] + string[2] + string[3] + "/" + string[4] + string[5] + "/" + string[6] + string[7]

    return datetime.datetime.strptime(new_string, "%Y/%m/%d") 


"""
plot positivity
"""

def plot_positivity_persons(date, dico_score, score_moyen):

    list_score = list()
    list_names = list()

    for name, score in dico_score.items():
        list_score.append(score)
        list_names.append(name)


    plt.bar(list_names, list_score, color='r')
    plt.grid()
    plt.xticks(rotation=90)
    plt.text(-1, 0, date, horizontalalignment='left', verticalalignment='top', fontweight='bold')
    plt.hlines(y=score_moyen, xmin = -1, xmax = len(list_names))
    plt.ylabel("Score de positivité")
    plt.title("Score de positivité des principaux speakers")
    plt.tight_layout()
    #plt.show()
    plt.savefig(project_directory + 'image_score_posi/' + 'score_posi_' + str(date) + '.png')
    plt.close()

    return None



"""
Main
"""




for date in l_dates_final[-50:]:

	with open (project_directory+'sentences_by_names/'+str(date)+'meeting.txt', 'r') as doc:
		content = doc.readlines()[0]
			    
	dictionary = ast.literal_eval(content)

	#Cleaning 
	dico_clean = remove_stop_word(clean_dico_new_line(dictionary))
	plot_positivity_persons(date, compute_positivity(dico_clean), compute_mean_positivity(dico_clean))

	
	
























