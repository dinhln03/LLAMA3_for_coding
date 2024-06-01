# -*- coding: utf-8 -*-
# This script was written by Takashi SUGA on April-August 2017
# You may use and/or modify this file according to the license described in the MIT LICENSE.txt file https://raw.githubusercontent.com/suchowan/watson-api-client/master
"""『重要文抽出によるWebページ要約のためのHTMLテキスト分割』
    http://harp.lib.hiroshima-u.ac.jp/hiroshima-cu/metadata/5532
    を参考にした HTML テキスト化処理
"""
import codecs
import re

class Article:

    # この順に文字コードを試みる
    encodings = [
        "utf-8",
        "cp932",
        "euc-jp",
        "iso-2022-jp",
        "latin_1"
    ]

    # ブロックレベル要素抽出正規表現
    block_level_tags = re.compile("(?i)</?(" + "|".join([
        "address", "blockquote", "center", "dir", "div", "dl",
        "fieldset", "form", "h[1-6]", "hr", "isindex", "menu",
        "noframes", "noscript", "ol", "pre", "p", "table", "ul",
        "dd", "dt", "frameset", "li", "tbody", "td", "tfoot",
        "th", "thead", "tr"
        ]) + ")(>|[^a-z].*?>)")

    def __init__(self, path):
        print(path)
        self.path = path
        self.contents = self.get_contents()
      # self.contents = self.get_title()

    def get_contents(self):
        for encoding in self.encodings:
            try:
                lines = codecs.open(self.path, 'r', encoding)
                html = ' '.join(line.rstrip('\r\n') for line in lines)
                return self.__get_contents_in_html(html)
            except UnicodeDecodeError:
                continue
        print('Cannot detect encoding of ' + self.path)
        return None

    def __get_contents_in_html(self, html):
        parts = re.split("(?i)<(?:body|frame).*?>", html, 1)
        if len(parts) == 2:
            head, body = parts
        else:
            print('Cannot split ' + self.path)
            body = html
        body = re.sub(r"(?i)<(script|style|select).*?>.*?</\1\s*>", " ", body)
        body = re.sub(self.block_level_tags, ' _BLOCK_LEVEL_TAG_ ', body)
        body = re.sub(r"(?i)<a\s.+?>", ' _ANCHOR_LEFT_TAG_ ', body)
        body = re.sub("(?i)</a>", ' _ANCHOR_RIGHT_TAG_ ', body)
        body = re.sub("(?i)<[/a-z].*?>", " ", body)
        return re.sub(" +", " ", "".join(self.__get_contents_in_body(body)))

    def __get_contents_in_body(self, body):
        for block in body.split("_BLOCK_LEVEL_TAG_"):
            yield from self.__get_contents_in_block(block)

    def __get_contents_in_block(self, block):
        self.in_sentence = False
        for unit in block.split("。"):
            yield from self.__get_contents_in_unit(unit)
        if self.in_sentence:
            yield '。\n'

    def __get_contents_in_unit(self, unit):
        image_link = "_ANCHOR_LEFT_TAG_ +_ANCHOR_RIGHT_TAG_"
        unit = re.sub(image_link, " ", unit)
        if re.match(r"^ *$", unit):
            return
        fragment_tag = "((?:_ANCHOR_LEFT_TAG_ .+?_ANCHOR_LEFT_TAG_ ){2,})"
        for fragment in re.split(fragment_tag, unit):
            yield from self.__get_contents_in_fragment(fragment)

    def __get_contents_in_fragment(self, fragment):
        fragment = re.sub("_ANCHOR_(LEFT|RIGHT)_TAG_", ' ', fragment)
        if re.match(r"^ *$", fragment):
            return
        text_unit = TextUnit(fragment)
        if text_unit.is_sentence():
            # 文ユニットは“ 。”で終わる
            if self.in_sentence:
                yield '。\n'
            yield text_unit.separated
            yield ' 。\n'
            self.in_sentence = False
        else:
            # 非文ユニットは“―。”で終わる
            # (制約) 論文と相違し非文ユニットは結合のみ行い分割していない
            yield text_unit.separated
            yield '―'
            self.in_sentence = True

    def get_title(self):
        return self.path.split('/')[-1]

from janome.tokenizer import Tokenizer
from collections import defaultdict
import mojimoji
#import re

class TextUnit:

    tokenizer = Tokenizer("user_dic.csv", udic_type="simpledic", udic_enc="utf8")

    def __init__(self,fragment):
        self.fragment   = fragment
        self.categories = defaultdict(int)
        separated  = []
        for token in self.tokenizer.tokenize(self.preprocess(self.fragment)):
            self.categories[self.categorize(token.part_of_speech)] += 1
            separated.append(token.surface)
        separated.append('')
        self.separated = '/'.join(separated)

    def categorize(self,part_of_speech):
        if re.match("^名詞,(一般|代名詞|固有名詞|サ変接続|[^,]+語幹)", part_of_speech):
            return '自立'
        if re.match("^動詞", part_of_speech) and not re.match("サ変", part_of_speech):
            return '自立'
        if re.match("^形容詞,自立", part_of_speech):
            return '自立'
        if re.match("^助詞", part_of_speech):
            return '助詞'
        if re.match("^助動詞", part_of_speech):
            return '助動詞'
        return 'その他'

    def is_sentence(self):
        if self.categories['自立'] == 0:
            return False
        match = 0
        if self.categories['自立'] >= 7:
            match += 1
        if 100 * self.categories['自立'] / sum(self.categories.values()) <= 64:
            match += 1
        if 100 * (self.categories['助詞'] + self.categories['助動詞']) / self.categories['自立'] >= 22:
            # 論文通り「付属語　＝ 助詞　⋃　助動詞」と解釈 (通常の定義と異なる)
            match += 1
        if 100 * self.categories['助詞'] / self.categories['自立'] >= 26:
            match += 1
        if 100 * self.categories['助動詞'] / self.categories['自立'] >= 6:
            match += 1
        return match >= 3

    def preprocess(self, text):
        text = re.sub("&[^;]+;",  " ", text)
        text = mojimoji.han_to_zen(text, digit=False)
        text = re.sub('(\t |　)+', " ", text)
        return text

if __name__ == '__main__':
    import glob
    import os

    path_pattern = '/home/samba/example/links/bookmarks.crawled/**/*.html'
    # The converted plaintext is put as '/home/samba/example/links/bookmarks.plaintext/**/*.txt'
    for path in glob.glob(path_pattern, recursive=True):
        article = Article(path)
        plaintext_path = re.sub("(?i)html?$", "txt", path.replace('.crawled', '.plaintext'))
        plaintext_path = plaintext_path.replace('\\', '/')
        plaintext_dir  = re.sub("/[^/]+$", "", plaintext_path)
        if not os.path.exists(plaintext_dir):
            os.makedirs(plaintext_dir)
        with codecs.open(plaintext_path, 'w', 'utf-8') as f:
            f.write(article.contents)
