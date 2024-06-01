import re

from chatterbot.conversation import Statement

from chatbot.const import const
from chatbot.vocabulary import Word


class Detector(object):
    def __init__(self):
        self.vocabulary = Word()
        self.enum_type_key_word = ('类型', '等级', '方式', '分类', '模式', 'type', 'class', '系列')
        self.brand_type_key_word = ('品牌', '产品')
        self.text_type_key_word = ('简介', '描述', '简称', '备注', '说明',)
        self.date_type_key_word = ('日期', '时间', '日', '年', '月',)
        self.person_type_key_word = ('创办人', '负责人', '经理', '经手人', '经办人')
        self.org_type_key_word = ('托管方', '保管方',)
        self.price_type_key_word = ('价格', '金额', '价', '额度', '利润', '收益', '成本', '支出')
        self.mass_type_key_word = ('重量', '毛重', '净重', '毛重量', '净重',)
        self.volume_type_key_word = ('体积', '容量', '大小')
        self.length_type_key_word = ('长度', '宽度', '高度', '长', '宽', '高')

        self.operation_pattern = const.COMPARISON_PATTERN

    def detect_type_column(self, col_name) -> str:
        seg_words = self.vocabulary.get_seg_words(col_name)

        last_word = str(seg_words[-1]).lower()

        if last_word in self.enum_type_key_word:
            return const.ENUM

        if last_word in self.brand_type_key_word:
            return const.BRAND

        if last_word in self.date_type_key_word:
            return const.DATE

        if last_word in self.person_type_key_word:
            return const.PERSON

        if last_word in self.org_type_key_word:
            return const.ORG

        if last_word in self.price_type_key_word:
            return const.PRICE

        return const.TEXT

    def detect_operation(self, statement: Statement):
        query_text = statement.text
        seg_word = statement.search_text.split(const.SEG_SEPARATOR)
        operation = {}

        phrase = []

        for op in self.operation_pattern.keys():
            for pattern, slot_type, word, unit in self.operation_pattern[op]:
                match = re.search(pattern, query_text)
                if match:
                    operation['op'] = op
                    operation['slot_type'] = slot_type

                    words = match.groups()[0]

                    for w in seg_word:
                        if w in words:
                            phrase.append(w)
                    operation['phrase'] = phrase
                    operation['word'] = word
                    operation['unit'] = unit
                    return operation

        return operation
