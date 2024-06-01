#!/usr/bin/env python3


"""
Possible string formats:
<author(s)> <title> <source> <year>
"""


import re
import pdf


CRED = '\033[91m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'
CBLUE = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE = '\33[36m'
CWHITE = '\33[37m'
CEND = '\033[0m'


def extract_references_list_by_keyword(text, keyword):
    print(text)
    match_res = re.search(keyword, text)
    ref_text = text[match_res.span()[0]:]
    # print(ref_text)
    # WARNING: not more than 999 references!
    index_re = re.compile('\[[0-9]([0-9]|)([0-9]|)\]')
    ref_pos = []
    for ref in index_re.finditer(ref_text):
        ref_pos.append(ref.span()[0])
    ref_pos.append(len(ref_text))
    for i in range(len(ref_pos)-1):
        print(CYELLOW + ref_text[ref_pos[i]:ref_pos[i+1]] + CEND)


def extract_references_list(text):
    res = []

    buffer = ""
    state = 0
    for i in reversed(range(0, len(text)-1)):
        c = text[i]
        buffer = c + buffer
        if state == 0:
            if c == ']':
                state = 1
        elif state == 1:
            if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                state = 2
            else:
                state = 0
        elif state == 2:
            if c == '[':
                res.append(buffer)
                if buffer[1] == '1' and buffer[2] == ']':
                    break
                state = 0
                buffer = ""
        else:
            print("Unknown state")
            raise

    return reversed(res)


def extract_article_from_reference(string):
    pass
    # return (autors, title, date)


if __name__ == '__main__':
    import sys
    text = pdf.extract_text(sys.argv[1])
    print(text)
    # zextract_references_list_by_keyword('REFERENCES')
    ref_list = extract_references_list(text)
    for ref in ref_list:
        print(CYELLOW + ref + CEND)
