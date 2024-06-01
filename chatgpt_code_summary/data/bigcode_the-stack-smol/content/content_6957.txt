import string
from app.chatterbot import languages
from spacy.lang.zh import Chinese


class LowercaseTagger(object):
    """
    Returns the text in lowercase.
    """

    def __init__(self, language=None):
        self.language = language or languages.ENG

    def get_text_index_string(self, text):
        return text.lower()


class PosLemmaTagger(object):

    def __init__(self, language=None):
        import spacy
        self.language = language or languages.ENG
        punc = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        # punc = punc.decode("utf-8")
        self.punctuation_table = str.maketrans(dict.fromkeys(string.punctuation + punc))
        language = self.language.ISO_639_1.lower()
        if language == 'zh':
            self.nlp = Chinese()
        else:
            self.nlp = spacy.load(language)

    def get_text_index_string(self, text):
        """
        Return a string of text containing part-of-speech, lemma pairs.
        """

        if len(text) <= 2:
            text_without_punctuation = text.translate(self.punctuation_table)
            if len(text_without_punctuation) >= 1:
                text = text_without_punctuation
        document = self.nlp(text)
        if len(text) <= 2:
            bigram_pairs = [
                token.lemma_.lower() for token in document
            ]
            tokens = [ele for ele in bigram_pairs]
        else:
            tokens = [
                token for token in document if token.is_alpha and not token.is_stop
            ]

            if len(tokens) < 2:
                tokens = [
                    token for token in document if token.is_alpha
                ]
            tokens = [token.lemma_.lower() for token in tokens]
            # if not bigram_pairs:
            #     bigram_pairs = [
            #         token.lemma_.lower() for token in document
            #     ]
            #
            # return ' '.join(bigram_pairs)
        return ' '.join(tokens)


