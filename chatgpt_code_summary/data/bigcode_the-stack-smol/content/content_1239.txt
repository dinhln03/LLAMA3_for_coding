import pycrfsuite
import sklearn
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import re
import json

annotypes = ['Participants', 'Intervention', 'Outcome']
annotype = annotypes[0]
path = '/nlp/data/romap/crf/'
#path = '/Users/romapatel/Desktop/crf/'

def run():
    train_sents, test_sents = get_train_test_sets()
    print len(test_sents)
    indwords_list = get_ind_words()
    patterns_list = get_patterns()
    X_train = [sent_features(train_sents[docid], indwords_list, patterns_list) for docid in train_sents.keys()]
    y_train = [sent_labels(train_sents[docid]) for docid in train_sents.keys()]

    X_test = [sent_features(test_sents[docid], indwords_list, patterns_list) for docid in test_sents.keys()]
    y_test = [sent_labels(test_sents[docid]) for docid in test_sents.keys()]

    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({'c1': 1.0,'c2': 1e-3, 'max_iterations': 50, 'feature.possible_transitions': True})
    trainer.train('PICO.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('PICO.crfsuite')

    get_results(test_sents, tagger, indwords_list, patterns_list)


def get_results(test_sents, tagger, indwords_list, patterns_list):
    f1 = open(path + 'sets/4/' + annotype + '-test_pred.json', 'w+')
    f2 = open(path + 'sets/4/' + annotype + '-test_correct.json', 'w+')
    pred_dict, correct_dict = {}, {}
    for docid in test_sents:
        pred, correct = tagger.tag(sent_features(test_sents[docid], indwords_list, patterns_list)), sent_labels(test_sents[docid])
        spans, span, outside = [], [], True
        for i in range(len(pred)):
            if pred[i] == '0' and outside is True: continue
            elif pred[i] == '0' and outside is False:
                span.append(i+1)
                spans.append(span)
                span, outside = [], True
            elif pred[i] == '1' and outside is False: continue
            elif pred[i] == '1' and outside is True:
                outside = False
                span.append(i)
        pred_dict[docid] = spans
        
        spans, span, outside = [], [], True
        for i in range(len(correct)):
            if correct[i] == '0' and outside is True: continue
            elif correct[i] == '0' and outside is False:
                span.append(i+1)
                spans.append(span)
                span, outside = [], True
            elif correct[i] == '1' and outside is False: continue
            elif correct[i] == '1' and outside is True:
                outside = False
                span.append(i)
        correct_dict[docid] = spans
    f1.write(json.dumps(pred_dict))
    f2.write(json.dumps(correct_dict))                
                
def get_ind_words():
    fin_list = []
    for annotype in annotypes:
        list = []
        #filename = annotype.lower() + '_words.txt'
        filename = annotype.lower() + '_unigrams.tsv'
        f = open(path + 'crf_files/' + filename, 'r')
        for line in f:
            #word = line[:-1]
            items = line.split('\t')
            word = items[1][:-1]
            if word not in list:
                list.append(word)
        if annotype == 'Intervention':
            f = open(path + 'crf_files/drug_names.txt', 'r')
            for line in f:
                word = line[:-1]
                if word not in list:
                    list.append(word)
        fin_list.append(list)
    indwords = [fin_list[0], fin_list[1], fin_list[2]]
    return indwords

#all lowercased
def get_patterns():
    fin_list = []
    for annotype in annotypes:
        list = []
        #filename = annotype.lower() + '_pattern_copy.txt'
        filename = annotype.lower() + '_trigrams3.tsv'
        f = open(path + 'crf_files/' + filename, 'r')
        for line in f:
            #word = line[:-1].lower()
            word = line[:-1].split('\t')
            word = word[1]
            if word not in list:
                list.append(word)
        fin_list.append(list)
    patterns = [fin_list[0], fin_list[1], fin_list[2]]
    return patterns
    
def isindword(word, annotype, indwords_list):
    if annotype == annotypes[0]: list = indwords_list[0]
    elif annotype == annotypes[1]: list = indwords_list[1]
    else: list = indwords_list[2]
    f = open(path + 'crf_files/numbers.txt', 'r')
    for line in f:
        if line[:-1] in word.lower():
            return True
    if word.lower() in list or word.lower()[:-1] in list or word.lower()[-3:] in list: return True
    else: return False

def ispattern(word, pos, annotype, pattern_list):
    if annotype == annotypes[0]: list = pattern_list[0]
    elif annotype == annotypes[1]: list = pattern_list[1]
    else: list = pattern_list[2]
    for pattern in pattern_list:
        if word.lower() in pattern or pos.lower() in pattern: return True
    else: return False

def word_features(sent, i, indwords_list, pattern_list):
    word = sent[i][0]
    postag = sent[i][2]
    features = ['bias', 'word.lower=' + word.lower(),'word[-3:]=' + word[-3:],
        'word[-4:]=' + word[-4:],'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(), 'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag, 'isindword=%s' % isindword(word, annotype, indwords_list),
        'word[0:4]=' + word[0:4], 'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)]
    #prev previous word
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][2]

        features.extend(['-1:word.lower=' + word1.lower(), '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
            'word[-3:]=' + word[-3:]])
    #previous word
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][2]

        features.extend(['-1:word.lower=' + word1.lower(), '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
            'word[-3:]=' + word[-3:], 'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)])
    else:
        features.append('BOS')
    #next to next word
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][2]

        features.extend(['+1:word.lower=' + word1.lower(), '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(), '+1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
            'word[-3:]=' + word[-3:]])
    #next word
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][2]

        features.extend(['+1:word.lower=' + word1.lower(), '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(), '+1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
            'word[-3:]=' + word[-3:], 'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)])
    else:
        features.append('EOS')
    return features

def sent_features(sent, indwords_list, patterns_list):
    return [word_features(sent, i, indwords_list, patterns_list) for i in range(len(sent))]

def sent_labels(sent):
    return [str(p_label) for token, ner, postag, p_label, i_label, o_label in sent]

def sent_tokens(sent):
    return [token for token, ner, postag, p_label, i_label, o_label in sent]

def print_results(example_sent, tagger, indwords_list, docid, dict):
    pred, correct = tagger.tag(sent_features(example_sent, indwords_list)), sent_labels(example_sent)
    spans, span, outside = [], [], True
    for i in range(len(pred)):
        if pred[i] == '0' and outside is True: continue
        elif pred[i] == '0' and outside is False:
            span.append(i+1)
            spans.append(span)
            span, outside = [], True
        elif pred[i] == '1' and outside is False: continue
        elif pred[i] == '1' and outside is True:
            outside = False
            span.append(i)
            
    f = open(path + annotype + '-test.json', 'w+')

    print '\n\nPredicted: ' + str(spans)
    for span in spans:
        s = ' '
        for i in range(span[0], span[1]):
            s += example_sent[i][0] + ' '
        print s
        
    spans, span, outside = [], [], True
    for i in range(len(correct)):
        if correct[i] == '0' and outside is True: continue
        elif correct[i] == '0' and outside is False:
            span.append(i+1)
            spans.append(span)
            span, outside = [], True
        elif correct[i] == '1' and outside is False: continue
        elif correct[i] == '1' and outside is True:
            outside = False
            span.append(i)
    print '\n\nCorrect: ' + str(spans)
    for span in spans:
        s = ' '
        for i in range(span[0], span[1]):
            s += example_sent[i][0] + ' '
        print s
    
def get_training_data():
    f = open(path + 'crf_files/difficulty_crf_mv.json', 'r')
    for line in f:
        dict = json.loads(line)
    return dict

def get_train_test_sets():
    test_docids = []
    f = open(path + 'crf_files/gold_docids.txt', 'r')
    for line in f:
        test_docids.append(line[:-1])
    doc_dict = get_training_data()
    test_sents, train_sents = {}, {}
    count = 0
    for docid in doc_dict:
        sents = doc_dict[docid]
        if len(sents) == 0: continue
        count += 1
        #if count >= 100: break
        if docid not in test_docids:
            train_sents[docid] = sents
        else:
            test_sents[docid] = sents

    f = open(path + 'difficulty_new.json', 'r')
    for line in f:
        doc_dict_new = json.loads(line)
    count = 1
    for docid in doc_dict_new:
        if docid in train_sents.keys(): continue
        if count < 9481:
            count += 1
            continue
        train_sents[docid] = doc_dict_new[docid]
        count += 1
    return train_sents, test_sents
    
if __name__ == '__main__':
    run()

