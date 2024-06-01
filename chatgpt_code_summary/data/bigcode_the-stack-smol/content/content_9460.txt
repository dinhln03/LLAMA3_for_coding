JONG_COMP = {
    'ㄱ': {
        'ㄱ': 'ㄲ',
        'ㅅ': 'ㄳ',
    },
    'ㄴ': {
        'ㅈ': 'ㄵ',
        'ㅎ': 'ㄶ',
    },
    'ㄹ': {
        'ㄱ': 'ㄺ',
        'ㅁ': 'ㄻ',
        'ㅂ': 'ㄼ',
        'ㅅ': 'ㄽ',
        'ㅌ': 'ㄾ',
        'ㅍ': 'ㄿ',
        'ㅎ': 'ㅀ',
    }
}

DEFAULT_COMPOSE_SEPARATOR = u'ᴥ'


################################################################################
# Hangul Automata functions by bluedisk@gmail.com
################################################################################


def decompose(text, latin_filter=True, separator=DEFAULT_COMPOSE_SEPARATOR):
    from . import letter

    result = ""

    for c in list(text):
        if letter.is_hangul(c):
            result += "".join(letter.decompose(c)) + separator
        else:
            result = result + c

    return result


STATUS_CHO = 0
STATUS_JOONG = 1
STATUS_JONG1 = 2
STATUS_JONG2 = 3


def compose(text, compose_code=DEFAULT_COMPOSE_SEPARATOR):
    from .const import ONSET, NUCLEUS, CODA
    from . import letter

    res_text = ""

    status = STATUS_CHO

    for c in text:
        if status == STATUS_CHO:
            if c in ONSET:
                chosung = c
                status = STATUS_JOONG
            else:
                if c != compose_code:
                    res_text = res_text + c

        elif status == STATUS_JOONG:

            if c != compose_code and c in NUCLEUS:
                joongsung = c
                status = STATUS_JONG1
            else:
                res_text = res_text + chosung

                if c in ONSET:
                    chosung = c
                    status = STATUS_JOONG
                else:
                    if c != compose_code:
                        res_text = res_text + c
                    status = STATUS_CHO

        elif status == STATUS_JONG1:

            if c != compose_code and c in CODA:
                jongsung = c

                if c in JONG_COMP:
                    status = STATUS_JONG2
                else:
                    res_text = res_text + letter.compose(chosung, joongsung, jongsung)
                    status = STATUS_CHO

            else:
                res_text = res_text + letter.compose(chosung, joongsung)

                if c in ONSET:
                    chosung = c
                    status = STATUS_JOONG
                else:
                    if c != compose_code:
                        res_text = res_text + c

                    status = STATUS_CHO

        elif status == STATUS_JONG2:

            if c != compose_code and c in JONG_COMP[jongsung]:
                jongsung = JONG_COMP[jongsung][c]
                c = compose_code  # 종성째 출력 방지

            res_text = res_text + letter.compose(chosung, joongsung, jongsung)

            if c != compose_code:
                res_text = res_text + c

            status = STATUS_CHO

    return res_text
