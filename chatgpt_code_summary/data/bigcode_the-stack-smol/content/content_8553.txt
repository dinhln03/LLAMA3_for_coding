#!/usr/bin/env python

from markovLatex.markov import TextGenerator
import argparse, sys

def main(args):

    parser = argparse.ArgumentParser(description="Generate pseudorandom text from input text files.")
    parser.add_argument('files', type=str, nargs='+', help='one or more input text files')
    parser.add_argument('-p','--paragraphs', type=int, default=1, help='The number of paragraphs in the output text')
    args = parser.parse_args(args)

    corpus = []
    for arg in args.files:
        try:
            corpus.append(open(arg, 'r'))
        except Exception as e:
            print(e)

    if corpus:
        try:
            text_generator = TextGenerator(corpus)
            for i in range(0, args.paragraphs):
                print(text_generator.paragraph())
        
        except Exception as e:
            print(e)
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
