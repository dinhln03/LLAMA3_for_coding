'''
@author: kris
'''
# import modules; set up logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np
import logging, os, sys, gzip
import datetime


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename='word2vec.out', level=logging.INFO)
# Path to a file that contains lines with the locations of files 
# containing the sentences we want for our Word2Vec model
# Also works with entities that are just stacked line by line
pathsLocator = "./sentencesPaths.txt"
outputPath = "./entity_embeddings.txt"

# Model to load
to_load = '/vol2/cb/crunchbase-201806/embeddings/dim200-iter10-win5/CB_sg1_size200_mincount1_window5_neg15_iter10.wv.vectors.npy'
#'/home/faerberm/mag-training/MAG_sg1_size128_minCount5_window5_neg15_iter10_alpha_cbowMean.wv.vectors.npy'
#'/vol2/cb/crunchbase-201806/embeddings/dim200-iter10-win5/CB_sg1_size200_mincount1_window5_neg15_iter10'
#'MAG_sg1_size128_minCount5_window5_neg15_iter5'
loadKeyedVector = True

#'dbpedia_sg1_size200_mincount1_window5_neg15_iter10'
#'RDF2Vec_sg1_size200_mincount1_window5_neg15_iter20'
#'MAG_sg1_size200_mincount1_window5_neg15_iter15'

#What is the newline character on the machine
newline = '\n'
ignorePrefix = '#'
#What separates one walk from another (aka. one sentence from another)?
walkSeparator = "\t"
#What separates the single 'units' of a given walk?
hopSeparator = '->'

# Mapping dict
entity_mapping_dict = {}
# Mapping file
mapping_file = "/home/noulletk/prog/bmw/dbpedia_full/resources/data/walks/walk_entity_mapping.txt"
mapping_sep = "\t"
hasMapping = False

iterationCounter = {'val': 0}

#Load mappings if there are any
if hasMapping:
	for mapping_line in open(mapping_file, mode='rt'):
		mapping_tokens = mapping_line.rstrip(newline).split(mapping_sep)
		if len(mapping_tokens) == 2:
			entity_mapping_dict[mapping_tokens[0]] = mapping_tokens[1]
	print("Loaded %s mappings!" % (len(entity_mapping_dict)))


class MySentences:
	def __init__(self, iterationCounter):
		self.iterationCounter = iterationCounter

	def __iter__(self):
		print("Running Iteration #%s" % (iterationCounter['val']))
		iterationCounter['val'] += 1
		# Iterate to find which files are to be read
		for fname in open(pathsLocator, mode='rt'):  # os.listdir(self.dirname):
			sentencesPath = fname.rstrip(newline)
			# Ignore commented-out lines
			if sentencesPath.startswith(ignorePrefix):
				continue
			now = datetime.datetime.now()
			print("[%s] Grabbing sentences from: %s" % (now.strftime("%Y-%m-%d %H:%M"), sentencesPath))
			try:
				# Go through all paths
				for line in open(sentencesPath, mode='rt'):
					# If you're NOT grouping the walks and separating them by tabs
					sentence = line.rstrip(newline).split(hopSeparator)
					for tokenPos in range(len(sentence)):
						token = sentence[tokenPos]
						# Give the proper URL for the entity IF it exists, otherwise return the entity itself
						sentence[tokenPos] = entity_mapping_dict.get(token, token)
					#print(sentence)
					yield sentence
						
			except Exception:
				print("Failed reading file:")
				print(sentencesPath)

#load model
if loadKeyedVector:
	print("Loading [KeyedVectors] from: ",to_load)
	#model_wv = KeyedVectors.load(to_load, mmap='r')
	#model_wv = KeyedVectors.load_word2vec_format(to_load, binary=True)
	#model_wv = KeyedVectors.load_word2vec_format(to_load)
	model_wv = KeyedVectors.load(to_load)
	#model_wv = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
	#model_wv = KeyedVectors.load_word2vec_format(to_load, binary=True, unicode_errors='ignore')   
else:
	print("Loading [MODEL] from: ",to_load)
	model_wv = Word2Vec.load(to_load).wv


print("Vocab keys size:",len(model_wv.vocab.keys()))
print("Outputting entity embeddings to: ",outputPath)
sentences = MySentences(iterationCounter)

#Open the output file for the entity embeddings
outFile = open(outputPath, "w")

#Make a dictionary for in-memory aggregation while going over sentences
default_val = None
entity_embeddings_dict = {}
vocab_keys = model_wv.vocab.keys()

displayCounter = 0
maxDisplay = 10
for voc in vocab_keys:
	print(voc)
	if displayCounter >= maxDisplay:
		break
	displayCounter+=1


print("Compute entity embeddings (through combination of word embeddings)...")
counter = 0

'''
for sentence in sentences:
	entity = sentence[0]
	entity_embedding = None
	#Sum over all words' embeddings and then output the resulting embedding
	for word in sentence:
		word_embedding = model.wv[word]
		if default_val is None:
			#Initialise default_val if it isn't yet
			default_val = np.zeros(word_embedding.shape)
		if entity_embedding is None:
			entity_embedding = np.zeros(word_embedding.shape)
		entity_embedding += word_embedding
	entity_embeddings_dict[entity] = entity_embeddings_dict.get(entity, default_val) + entity_embedding
	if (counter % 1000000 == 0):
		print("Combined word embeddings: ",counter)
		print("Last one completed: ",entity)
	counter+=1
'''
	
#Go through all sentences to see which entities we want
for sentence in sentences:
	# idea is that the entity is in the document, so we check what it is like and 
	# since every entity has 'the same' treatment, that we can determine their probabilities based on that
	entity = sentence[0]
	if hasMapping:
		entity = entity_mapping_dict.get(entity, entity)
	entity_embedding = None
	dict_val = entity_embeddings_dict.get(entity, None)
	if (dict_val is None):
		if entity in vocab_keys:
			entity_embedding = model_wv[entity]
			entity_embeddings_dict[entity] = entity_embedding
			#Encountered first time, so output it
			outFile.write("%s" % entity)
			for number in entity_embedding:
				outFile.write("\t%s" % number)
			outFile.write("\n")

	if (counter % 1000000 == 0):
		print("Lines passed through: ",counter)
		print("Current line's entity: ",entity)
		print("Embeddings output: ",len(entity_embeddings_dict))
	counter+=1


#print("Output computed entity embeddings!")
#for (entity, entity_embedding) in entity_embeddings_dict.items():
#	#Output computed embedding
#	outFile.write("%s" % entity)
#	for number in entity_embedding:
#		outFile.write("\t%s" % number)
#	outFile.write("\n")

#Close the output file post finishing output operations
outFile.close()
print("Finished outputting entity embeddings")
