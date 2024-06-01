import sys
import json

import scrapapps
import scrapping
from textrank import TextRankSentences
import preprocessing
import summ
import textrankkeyword
import bss4

url = sys.argv[1]

# url = request.POST.get('web_link', None)
#web_link = scrapapps.scrap_data(url)
web_link = scrapping.scrap_data(url)

#Get Title
judul = scrapping.get_title(url)

raw_text = str(web_link)

# Preprocessing View
lower = preprocessing.text_lowercase(str(web_link))
rnumber = preprocessing.remove_numbers(lower)
white_space = preprocessing.remove_whitespace(rnumber)
stopword_list = preprocessing.remove_stopwords(white_space)

new_sentence = ' '.join(stopword_list)
stagging = preprocessing.stagging_text(new_sentence)

stop_plus = preprocessing.stopword_plus(new_sentence)
kalimat = ' '.join(stop_plus)

# Skenario 1

# n = 10;
# if len(stagging) < 10:
# 	n = 5

# if len(stagging) == 10:
# 	n = len(stagging) - 2

# if len(stagging) > 30:
# 	n = 15

# if len(stagging) < 5:
# 	n = len(stagging) - 1

# if len(stagging) == 1:
# 	n = len(stagging)

# Skenario 2
n = 7
if len(stagging) < 7:
    n = len(stagging) - 1

if len(stagging) == 1:
    n = len(stagging)

textrank = TextRankSentences()
text = textrank.analyze(str(new_sentence))
text = textrank.get_top_sentences(n)


# View Similarity Matriks
sim_mat = textrank._build_similarity_matrix(stagging)

#View Hasil Perhitungan Textrank
top_rank = textrank._run_page_rank(sim_mat)

result = textrank._run_page_rank(sim_mat)

# Clean Hasil
ringkasan = preprocessing.remove_punctuation(text)



# Panjang Plaintext
len_raw = len(str(web_link))

# Jumlah Text
len_text = len(str(text))

# Jumlah Kalimat
len_kalimat = len(stagging)

#Presentase Reduce
presentase = round(((len_text/len_raw)*100))


# keyphrases = textrankkeyword.extract_key_phrases(raw_text)

data = {
    'raw_text' : raw_text,
    'url' : url,
    'judul' : judul,
    'ringkasan':ringkasan,
    'text':text,
    'len_raw':len_raw,
    'len_text':len_text,
    'len_kalimat':len_kalimat,
    'stagging':stagging,
    'new_sentence':new_sentence,
    # 'sim_mat':sim_mat,
    # 'result':result,
    'presentase':presentase,
    'keyword':'-',
}

print(json.dumps(data))