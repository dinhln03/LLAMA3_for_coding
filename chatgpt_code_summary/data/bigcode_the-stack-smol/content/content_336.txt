import spacy
from spacy.tokens import Doc, Span, Token
import urllib
import xml.etree.ElementTree as ET
import re
from SpacyHu.BaseSpacyHuComponent import BaseSpacyHuComponent


class HuLemmaMorph(BaseSpacyHuComponent):
    def __init__(self,
                 nlp,
                 label='Morph',
                 url='http://hlt.bme.hu/chatbot/gate/process?run='):
        necessary_modules = ['QT', 'HFSTLemm']
        super().__init__(nlp, label, url, necessary_modules)
        Token.set_extension('morph', default='')
        Token.set_extension('lemma', default='')

    def get_word_from_annotation(self, annotation):
        for feature in annotation.getchildren():
            if feature.find('Name').text == 'string':
                return feature.find('Value').text

    def get_token_by_idx(self, idx, doc):
        for token in doc:
            if token.idx == idx:
                return token

    def get_lemma_from_morph(self, morph):
        return set(re.findall(r'(?<=lemma=).*?(?=\})', morph))

    def __call__(self, doc):
        text = urllib.parse.quote_plus(doc.text)
        result = urllib.request.urlopen(self.url + text).read()
        annotationset = ET.fromstring(result).find('AnnotationSet')
        for annotation in annotationset.getchildren():
            if annotation.get('Type') != 'Token':
                continue

            word_index = int(annotation.get('StartNode'))
            word = self.get_word_from_annotation(annotation)
            for feature in annotation.getchildren():
                if feature.find('Name').text == 'anas':
                    token = self.get_token_by_idx(word_index, doc)
                    anas = (feature.find('Value').text
                            if feature.find('Value').text is not None
                            else '')
                    token._.morph = set(anas.split(';'))
                    token._.lemma = self.get_lemma_from_morph(anas)
                    break

        return doc


if __name__ == "__main__":
    from Tokenizer import HuTokenizer

    debug_text = 'Jó, hogy ez az alma piros, mert az olyan almákat szeretem.'
    # debug_text = 'megszentségteleníthetetlenségeitekért meghalnak'
    remote_url = 'http://hlt.bme.hu/chatbot/gate/process?run='
    nlp = spacy.blank("en")
    nlp.tokenizer = HuTokenizer(nlp.vocab, url=remote_url)
    morph_analyzer = HuLemmaMorph(nlp, url=remote_url)
    nlp.add_pipe(morph_analyzer, last=True)

    doc = nlp(debug_text)
    for token in doc:
        print('Token is: ' + token.text)
        print(token._.lemma)
        print(token._.morph)
        print()
