# Usage: testWordsInCorpus.py [language] {corpus file}
# If no corpus file is named, the programme will try to load a corresponding cPickle file.
#
# German corpus: /mounts/data/proj/huiming/SIGMORPHON/dewiki-20151102-pages-articles-multistream.xml
#
# This script finds words that should belong to a paradigm in the corpus and adds them (for training?).

from getEditTrees import editTreesByPos
from getEditTrees import applyOnlyTree
import sys
import pickle as cPickle


toAdd = {} # lemma to things that should be autocompleted
uniquenessCheck = {} # (lemma, form) -> word, avoiding that we add things we are unsure about


# New autocomplete. Finds union and checks if paradigms can complete each other.
# We suppose the union consists of at least 2 edit trees.
# TODO: account for Umlaute.
# Returns a dictinary lemma -> (et, tags) with things to add to the original one.

# TODO: irgendwas stimmt hier nicht. korrigiere es
def autoComplete(lemma1, etTag1, lemma2, etTag2, corpusWords):
  etAndTagToAdd = set()
  notFound = 0
  allRight1 = True
  allRight2 = True
  
  for (et, form) in etTag1.difference(etTag2):
    
    result = applyOnlyTree(lemma2, et)

    if result == '#error#':
      allRight = False
      break
    if result not in corpusWords or corpusWords[result] <=3: # orig is 3
      notFound += 1
      if notFound == 2:
        allRight = False
        break
    else:
      etAndTagToAdd.add((et, form)) 

    
  if allRight and etAndTagToAdd:
    if lemma2 not in toAdd:
      toAdd[lemma2] = set()
    toAdd[lemma2] = toAdd[lemma2].union(etAndTagToAdd)
    for (et, form) in etAndTagToAdd:
      if (lemma2, form) not in uniquenessCheck:
        uniquenessCheck[(lemma2, form)] = set()
      else:
        if applyOnlyTree(lemma2,et) not in uniquenessCheck[(lemma2, form)]:
          print("yeay")
      uniquenessCheck[(lemma2, form)].add(applyOnlyTree(lemma2, et))
      
      
# Lemma 1 has more ETs than lemma 2.
# Returns a dictinary lemma -> (et, tags) with things to add to the original one.
def autoComplete2(lemma1, etTag1, lemma2, etTag2, corpusWords):
  etAndTagToAdd = set()
  notFound = 0
  allRight = True
  
  for (et, form) in etTag1.difference(etTag2):
    
    result = applyOnlyTree(lemma2, et)

    if result == '#error#':
      allRight = False
      break
    if result not in corpusWords or corpusWords[result] <=3: # orig is 3
      notFound += 1
      if notFound == 2:
        allRight = False
        break
    else:
      etAndTagToAdd.add((et, form)) 

    
  if allRight and etAndTagToAdd:
    if lemma2 not in toAdd:
      toAdd[lemma2] = set()
    toAdd[lemma2] = toAdd[lemma2].union(etAndTagToAdd)
    for (et, form) in etAndTagToAdd:
      if (lemma2, form) not in uniquenessCheck:
        uniquenessCheck[(lemma2, form)] = set()
      uniquenessCheck[(lemma2, form)].add(applyOnlyTree(lemma2, et))
  

# Test if a group of (edit tree, tag) combinations for a lemma is subset of the one for another lemma.
# If yes, try if the missing edit trees are applicable and if the corresponding word appears in the corpus.
def getAdditionalWords(lemmaToEtAndTag, corpusWords):
  isTrue = 0
  isFalse = 0
  for lemma1, etTag1 in lemmaToEtAndTag.items():
    for lemma2, etTag2 in lemmaToEtAndTag.items():
      if len(etTag1) <= 1 or len(etTag2) <= 1: # for now, don't complete things with 0 or only 1 entry. We are just not sure enough.
        isFalse += 1
        continue
      maybeSame = False
      if len(etTag1) > len(etTag2)+2:
        if len(etTag1) >= 3 and len(etTag2.union(etTag1)) > 1 and etTag2.issubset(etTag1):
          maybeSame = True
          autoComplete(lemma1, etTag1, lemma2, etTag2, corpusWords)
          isTrue += 1
        else:
          isFalse += 1
      elif len(etTag2) > len(etTag1)+2:
        if len(etTag2) >= 3 and len(etTag2.union(etTag1)) > 1 and etTag1.issubset(etTag2):
          maybeSame = True
          autoComplete(lemma2, etTag2, lemma1, etTag1, corpusWords)
          isTrue += 1
        else:
          isFalse += 1
         
  #print(str(len(toAdd)) +  ' words have been added.')
  #print("Is subset: " + str(isTrue))
  #print("No subset: " + str(isFalse))
  #sys.exit(0)
  noWordsToAdd = 0
  for lemma, aSet in toAdd.items():
    noWordsToAdd += len(aSet)
    
  '''
  for (lemma, form), word in uniquenessCheck.items():
    if len(word) > 1:
      print(word)
  
  sys.exit(0)
  '''
  return noWordsToAdd

def announce(*objs):
    print("# ", *objs, file = sys.stderr)

if __name__ == "__main__":
  lang = sys.argv[1]
  if len(sys.argv) == 2:
    usePickle = True
  else:
    usePickle = False
    
  posToEt, lemmaToEtAndTag = editTreesByPos(lang)
  
  for lemma, aSet in lemmaToEtAndTag.items():
    for (et, form) in aSet:
      if (lemma, form) not in uniquenessCheck:
        uniquenessCheck[(lemma, form)] = set()
      uniquenessCheck[(lemma, form)].add(applyOnlyTree(lemma, et))
      #print(applyOnlyTree(lemma, et))
  #sys.exit(0)

  if not usePickle:
    # Read the bonus corpus.
    announce('Start reading corpus...')
    corpusWords = {} # word to its frequency
    with open(sys.argv[2], 'r') as corpus_file:
      for line in corpus_file:
        #tokens = tokenize.word_tokenize(line.strip())
        tokens = line.strip().split(' ')
        for token in tokens:
          if token not in corpusWords:
            corpusWords[token] = 0
          corpusWords[token] += 1
    announce('Done reading corpus.')
    # Store the dictionary to a binary file.
    print('Store the dictionary with the corpus words to a binary file...')
    save_file = open('/mounts/data/proj/huiming/SIGMORPHON/corpusWords_' + lang, 'wb')
    cPickle.dump(corpusWords, save_file, -1)
    save_file.close()
    print('Done.')
  else:
    # Load the corpusWords dictionary.
    announce('Load the words with cPickle...')
    vocListFile = open('/mounts/data/proj/huiming/SIGMORPHON/corpusWords_' + lang, 'rb')
    corpusWords = cPickle.load(vocListFile)
    vocListFile.close()
    announce('Words loaded.')
  
  lastNumber = 0
  noWordsToAdd = 1
  while noWordsToAdd > lastNumber:
    lastNumber = noWordsToAdd
    noWordsToAdd = getAdditionalWords(lemmaToEtAndTag, corpusWords)
    
    for lemma, aSet in lemmaToEtAndTag.items():
      if lemma in toAdd:
        lemmaToEtAndTag[lemma] = lemmaToEtAndTag[lemma].union(toAdd[lemma])
    announce('Number word to add: ' + str(noWordsToAdd))
   
  # The union did not work well for some reason. Therefore, use toAdd directly.
  additionalWordsCounter = 0
  with open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + lang + '-bigger-task1-train', 'w') as out_file: 
    with open('/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/' + lang + '-task1-train', 'r') as original_file:
      for line in original_file:
        out_file.write(line)
    for lemma, etAndTagSet in toAdd.items():
      for (et, form) in etAndTagSet:
        if len(uniquenessCheck[(lemma, form)]) > 1:
          continue
        out_file.write(lemma + '\t' + form + '\t' + applyOnlyTree(lemma, et) + '\n')
        additionalWordsCounter += 1
        
  print(str(additionalWordsCounter) + ' words have been added.')
      