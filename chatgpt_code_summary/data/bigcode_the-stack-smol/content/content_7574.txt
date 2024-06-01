# -*- coding: utf-8 -*-
"""
Criado por Lucas Fonseca Lage em 04/03/2020
"""

import re, os, spacy
import numpy as np
from my_wsd import my_lesk
from unicodedata import normalize
from document import Document
from gensim.models import Phrases

# Carregamento do modelo Spacy
nlp = spacy.load('pt_core_news_lg')

# Carregamento dos modelos de bigramas e trigramas
#bigram_model = Phrases.load('./n_gram_models/bigram_gen_model')
#trigram_model = Phrases.load('./n_gram_models/trigram_gen_model')

freq_pos_tag = [('DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN'),
 ('VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN'),
 ('VERB', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT'),
 ('DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN', 'PUNCT'),
 ('NOUN', 'ADP', 'NOUN', 'ADP', 'DET', 'NOUN', 'PUNCT'),
 ('VERB', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN', 'PUNCT'),
 ('VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'DET'),
 ('DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN'),
 ('NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'NOUN', 'PUNCT'),
 ('VERB', 'DET', 'NOUN', 'ADP', 'NOUN', 'ADJ', 'PUNCT')]

def corpus_reader(path):
    '''Lê as extensões dos arquivos .xml no caminho especificado como path e
    retorna uma tupla com duas listas.Uma lista contém os paths para os arquivos
    .xml e a outra contém os arquivos Document gerados para aquele arquilo .xml
    '''
    prog = re.compile('(\.xml)$')
    doc_list = []

    f = []
    fps = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            fps.append(os.path.normpath(os.path.join(dirpath,filename)))

    for path in fps:
        if re.search(prog,path):
            f.append(path)
            doc_list.append(Document(path))
    return (f, doc_list)

def corpus_yeeter(path):
    '''Similar ao corpus_reader. Recebe um caminho para a pasta contendo o
     corpus e cria um generator. Cada iteração retorna uma tupla contendo um
    caminho para o arquivo .xml e o objeto Document criado a partir do mesmo
    '''
    prog = re.compile('(\.xml)$')
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if re.search(prog,filename):
                path = os.path.normpath(os.path.join(dirpath,filename))
                yield (path, Document(path))


def all_fps(path_to_dir):
    '''Recebe o caminho para o diretório e retorna uma lista com os caminhos
    absolutos para os arquivos que estão nele
    '''
    fps = []
    for dirpath, dirnames, filenames in os.walk(path_to_dir):
        for filename in filenames:
            fps.append(os.path.normpath(os.path.join(dirpath,filename)))
    return fps


def remover_acentos(text):
    '''Remove os acentos da string "text". Usada somente na função pre_process
    '''
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')


def pre_process(text):
    '''Realiza um pré processamento da string de entrada "text".
    Retira espaços em branco extras e retira caracteres não alfanuméricos
    '''
    text = re.sub('\s{2,}',' ',text).strip().lower()
    doc = nlp(text)
    #Retira numeros
    text = ' '.join([token.text for token in doc if token.is_alpha == True
                     and token.pos_ != 'PUNCT'])
    return remover_acentos(text)


def bi_trigram_counter(sentence_list):
    """Retorna uma tupla com o numero de bigramas e trigramas.
    Recebe como entrada o texto segmentado em uma lista de sentencas.
    """
    bi_sent_list = []
    tri_sent_list = []

    for sentence in sentence_list:
        proc_sent = pre_process(sentence).lower().split()
        bigram_sentence = bigram_model[proc_sent]
        bi_sent_list.append(bigram_sentence)

    for bi_sent in bi_sent_list:
        tri_sent = trigram_model[bi_sent]
        tri_sent_list.append(tri_sent)
    return(bigram_number(bi_sent_list),trigram_number(tri_sent_list))

def bigram_number(bigram_sent_list):
    '''Conta o número de bigramas encontrados na redação. Recebe uma lista de
    sentenças que configuram a redação.
    '''
    count = 0
    for sent in bigram_sent_list:
        for token in sent:
            if re.search(u'_',token):
                count += 1
    return count

def trigram_number(trigram_sent_list):
    '''Conta o número de trigramas encontrados na redação. Recebe uma lista de
    sentenças que configuram a redação
    '''
    count = 0
    for sent in trigram_sent_list:
        for token in sent:
            if re.search('(?<=_).+_',token):
                count += 1
    return count

def n_most_freq_pos_tag_seq(sent_list):
    ''' Procura na lista de sentenças a sequências de pos_tag mais frequentes e
    retorna a quantidade encontrada.
    '''
    n = 0
    pos_list = []

    for i in sent_list:
        sent_nlp = nlp(i)
        sent_pos = []
        for token in sent_nlp:
            sent_pos.append(token.pos_)
        pos_list.append(sent_pos)

    for line in pos_list:
        if len(line) < 7:
            continue
    if len(line) >= 7:
        while len(line) >= 7:
            t = tuple(line[0:7])
            if t in freq_pos_tag:
                n+=1
            line.pop(0)
    return n

def subj_n_elements(sentence_list):
    ''' Recebe a lista de sentenças da redação. Conta a quantidade de elementos
    abaixo do sujeito na árvore sintática gerada pelo "dependecy parser" do
    Spacy. Retorna o número de sujeitos que possuem uma quantidade de elementos
    maior que 7 e também o número total de elementos que fazem parte de um
    sujeito em toda a redação.
    '''
    r_list = []
    for spacy_doc in nlp.pipe(sentence_list):
        big_subj = 0
        subj_el_total = 0
        for token in spacy_doc:
            if token.dep_ == 'nsubj':
                size = len([desc for desc in token.subtree if desc.is_alpha])
                if size >= 7:
                    big_subj += 1
                subj_el_total += size
        r_list.append((big_subj,subj_el_total))
    return tuple([sum(i) for i in zip(*r_list)])

def synset_count(sent_list, lang='por', pos='NOUN'):
    i = 0
    for spacy_doc in nlp.pipe(sent_list):
        for token in spacy_doc:
            if token.pos_ == pos:
                i += len(wn.synsets(token.text, lang=lang))
    return (i, i/len(sent_list))

def hypo_hyper_count(sent_list):
    hyper = []
    hypo = []
    size = len(sent_list)
    for sent in nlp.pipe(sent_list):
        ss = [my_lesk(sent,token.text) for token in sent if token.pos_=='NOUN']
        for s in ss:
            try:
                hyper.append(len(s.hypernyms()))
                hypo.append(len(s.hyponyms()))
            except AttributeError:
                continue
    h_er_sum = sum(hyper)
    h_o_sum = sum(hypo)
    return(h_er_sum,h_er_sum/size, h_o_sum,h_o_sum/size)
