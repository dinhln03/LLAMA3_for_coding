#!/bin/python
"""
This is a class for loading input sentences
"""


class SentenceAttr:
    def __init__(self, attr_list):
        self.article_id = attr_list[1]
        self.title = attr_list[2]
        self.sentence = attr_list[3]
        self.article_structure = attr_list[4]
        self.place = attr_list[5]

    def __str__(self):
        return "Article Id: " + self.article_id + "\n" + "Title: " + self.title + "\n"\
                 +"Sentence: " + self.sentence + "\n" +\
                 "Article Structure: " + self.article_structure + "\n" + "Place: " + self.place + "\n"


class LoadSentences:
    def __init__(self, filepath, num):
        self.filepath = filepath
        self.num = num

    """对导入的文本做简单清洗"""

    def Process(self, line):
        line = line.replace('\n', '')
        line_list = line.split("|")
        return SentenceAttr(line_list)

    """逐行读取文件并返回迭代器"""

    def Reader(self):
        f = open(self.filepath)
        line = f.readline()
        count = 0
        while line:
            if count == self.num:
                break
            yield self.Process(line)
            line = f.readline()
            count += 1
        f.close()


def test():
    sentences_path = "../0_output.txt0.txt"
    sentences = LoadSentences(sentences_path, 5).Reader()
    for each in sentences:
        print(each)


if __name__ == "__main__":
    test()
