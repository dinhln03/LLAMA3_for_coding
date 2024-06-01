from lightning_conceptnet.uri import concept_uri
from wordfreq import simple_tokenize
from wordfreq.preprocess import preprocess_text


STOPWORDS = [
    'the', 'a', 'an'
]

DROP_FIRST = ['to']


def english_filter(tokens):
    """
    Given a list of tokens, remove a small list of English stopwords.
    """
    non_stopwords = [token for token in tokens if token not in STOPWORDS]
    while non_stopwords and non_stopwords[0] in DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    if non_stopwords:
        return non_stopwords
    else:
        return tokens


def standardized_concept_uri(lang, text, *more):
    """
    Make the appropriate URI for a concept in a particular language, including
    removing English stopwords, normalizing the text in a way appropriate
    to that language (using the text normalization from wordfreq), and joining
    its tokens with underscores in a concept URI.

    This text normalization can smooth over some writing differences: for
    example, it removes vowel points from Arabic words, and it transliterates
    Serbian written in the Cyrillic alphabet to the Latin alphabet so that it
    can match other words written in Latin letters.

    'more' contains information to distinguish word senses, such as a part
    of speech or a WordNet domain. The items in 'more' get lowercased and
    joined with underscores, but skip many of the other steps -- for example,
    they won't have stopwords removed.

    >>> standardized_concept_uri('en', 'this is a test')
    '/c/en/this_is_test'
    >>> standardized_concept_uri('en', 'this is a test', 'n', 'example phrase')
    '/c/en/this_is_test/n/example_phrase'
    >>> standardized_concept_uri('sh', 'симетрија')
    '/c/sh/simetrija'
    """
    lang = lang.lower()
    if lang == 'en':
        token_filter = english_filter
    else:
        token_filter = None

    text = preprocess_text(text.replace('_', ' '), lang)
    tokens = simple_tokenize(text)
    if token_filter is not None:
        tokens = token_filter(tokens)
    norm_text = '_'.join(tokens)
    more_text = []
    for item in more:
        if item is not None:
            tokens = simple_tokenize(item.replace('_', ' '))
            if token_filter is not None:
                tokens = token_filter(tokens)
            more_text.append('_'.join(tokens))

    return concept_uri(lang, norm_text, *more_text)
