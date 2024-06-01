import re

from haystack.inputs import Exact, Clean, BaseInput

from api.helpers.parse_helper import has_balanced_parentheses, matched_parens


class ElasticSearchExtendedAutoQuery(BaseInput):
    """
    A convenience class that handles common user queries.

    In addition to cleaning all tokens, it handles double quote bits as
    exact matches & terms with '-' in front as NOT queries.
    """
    input_type_name = 'auto_query'
    post_process = False
    exact_match_re = re.compile(r'"(?P<phrase>.*?)"')

    uncleaned_tokens = [
        'OR',
        'AND',
        'NOT',
        'TO',
    ]

    to_be_removed_special_chars_translation_table = {ord(c): None for c in matched_parens}

    def prepare(self, query_obj):
        query_string = super(ElasticSearchExtendedAutoQuery, self).prepare(query_obj)
        # Remove parens if they are not balanced
        if not has_balanced_parentheses(query_string):
            query_string = query_string.translate(self.to_be_removed_special_chars_translation_table)
        exacts = self.exact_match_re.findall(query_string)
        tokens = []
        query_bits = []

        for rough_token in self.exact_match_re.split(query_string):
            if not rough_token:
                continue
            elif rough_token not in exacts:
                # We have something that's not an exact match but may have more
                # than on word in it.
                tokens.extend(rough_token.split(' '))
            else:
                tokens.append(rough_token)
        for token in tokens:
            if not token:
                continue
            if token in exacts:
                query_bits.append(Exact(token, clean=True).prepare(query_obj))
            elif token in self.uncleaned_tokens:
                query_bits.append(token)
            else:
                query_bits.append(Clean(token).prepare(query_obj))
        return u' '.join(query_bits)
