from src.pre_processing import Preprocessing


def identifyQuery(query):
    q_l: str = query

    if q_l.__contains__("AND") or q_l.__contains__("OR") or q_l.__contains__("NOT"):
        return "B"
    elif query.__contains__("/"):
        return "PR"
    elif len(q_l.split()) == 1:
        return "S"
    else:
        return "PO"


def positionalSearch(query, stop_list, dict_book: dict):
    pipe = Preprocessing()
    pipe.stop_word = stop_list
    tokens = pipe.tokenizer(query)
    stems = pipe.stemmer(tokens)

    # dict_book structure: {"word": {doc-ID: [], ...}, ...}

    w1 = stems[0]
    w2 = stems[1]

    print(w1, w2)

    if dict_book.__contains__(w1) and dict_book.__contains__(w2):
        posting1: dict = dict_book.get(w1)  # dict returned, {docID:[], ...}
        posting2: dict = dict_book.get(w2)  # without using get() and type defining to be set not dict
    else:
        return []

    # len_posting1 = len(posting1)
    # len_posting2 = len(posting2)

    doc_list = []

    # i was iterating on sets rather than its keys
    for docI in posting1.keys():
        for docJ in posting2.keys():    # iterates on documents
            if docI == docJ:
                # print(docI)
                poslist1: list = posting1.get(docI)     # returns a position list
                poslist2: list = posting2.get(docJ)
                match: bool = False
                for pos1 in poslist1:  # hilary
                    for pos2 in poslist2:  # clinton
                        if pos2 - pos1 == 2:
                            doc_list.append(docI)
                            match = True
                            break
                    if match is True:
                        break
    return doc_list


def positionalSingleSearch(query, stop_list, dict_book: dict):
    pipe = Preprocessing()
    pipe.stop_word = stop_list
    tokens = pipe.tokenizer(query)
    stems = pipe.stemmer(tokens)

    # dict_book structure: {"word": {doc-ID: [], ...}, ...}

    w1 = stems[0]

    if dict_book.keys().__contains__(w1):
        posting1: dict = dict_book.get(w1)  # dict returned, {docID:[], ...}
        return list(posting1.keys())
    else:
        return []


def proximitySearch(query, stop_list, dict_book, k):
    pipe = Preprocessing()
    pipe.stop_word = stop_list
    tokens = pipe.tokenizer(query)
    stems = pipe.stemmer(tokens)

    w1 = stems[0]
    w2 = stems[1]

    if dict_book.__contains__(w1) and dict_book.__contains__(w2):
        posting1: dict = dict_book.get(w1)  # dict returned, {docID:[], ...}
        posting2: dict = dict_book.get(w2)
    else:
        return []

    # len_posting1 = len(posting1)
    # len_posting2 = len(posting2)

    doc_list = []

    for docI in posting1.keys():
        for docJ in posting2.keys():
            if docI == docJ:
                poslist1: list = posting1.get(docI)
                poslist2: list = posting2.get(docJ)
                match: bool = False
                for pos1 in poslist1:  # hilary
                    for pos2 in poslist2:  # clinton
                        if pos2 - pos1 == k+1:
                            doc_list.append(docI)
                            match = True
                            break
                    if match is True:
                        break
    return doc_list


def booleanSearch(query, stop_list, dict_book):
    pipe = Preprocessing()
    pipe.stop_word = stop_list
    # print(query)
    tokens = pipe.tokenizer(query)
    stems = pipe.stemmer(tokens)

    if dict_book.__contains__(stems):
        posting: set = dict_book[stems]
    else:
        return []

    doc_list = []

    for i in posting:
        doc_list.append(i)

    return doc_list
