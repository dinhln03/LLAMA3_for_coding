from gensim.models import word2vec

print("Learning Word2Vec embeddings")
tok_file = 'data/preprocessed/lemmatized.txt'
sentences = word2vec.LineSentence(tok_file)
model = word2vec.Word2Vec(sentences=sentences, size=10, window=5, workers=3, min_count=1)
model.wv.save_word2vec_format('models/vejica_word2vec.emb')

print("Saved Word2Vec format")