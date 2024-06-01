import optparse
import Utils
import gensim


def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--dataset', default='sample')
    parser.add_option('--size', default=300, type='int', help='vectors dimension. Default: %default')
    parser.add_option('--window', default=5, type='int', help='window size. Default: %default')
    parser.add_option('--min_count', default=5, type='int', help='Min count. Default: %default')
    options, args = parser.parse_args()

    documents = list(Utils.read_json('%s-tokenized.json' % options.dataset))
    model = gensim.models.word2vec.Word2Vec(documents, size=options.size, window=options.window, min_count=options.min_count, workers=4)

    model.save('%s-word-vector-model' % options.dataset)

main()    
