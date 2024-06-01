# custom PosLemmaTagger based on Chatterbot tagger

import string
from chatterbot import languages
import spacy

from chatterbot import tagging

class CustomPosLemmaTagger(tagging.PosLemmaTagger):

    def __init__(self, language=None):

        super(CustomPosLemmaTagger, self).__init__(language=None)

    def get_bigram_pair_string(self, text):
        """
        Return a string of text containing part-of-speech, lemma pairs.
        """
        bigram_pairs = []

        if len(text) <= 2:
            text_without_punctuation = text.translate(self.punctuation_table)
            if len(text_without_punctuation) >= 1:
                text = text_without_punctuation

        document = self.nlp(text)

        if len(text) <= 2:
            bigram_pairs = [
                token.lemma_.lower() for token in document
            ]
        else:
            tokens = [
                token for token in document if token.is_alpha and not token.is_stop
            ]

            if len(tokens) < 2:
                tokens = [
                    token for token in document if token.is_alpha
                ]

            for index in range(0, len(tokens)):
                bigram_pairs.append('{}:{}'.format(
                    tokens[index].pos_,
                    tokens[index].lemma_.lower()
                ))

        if not bigram_pairs:
            bigram_pairs = [
                token.lemma_.lower() for token in document
            ]

        return ' '.join(bigram_pairs)
        