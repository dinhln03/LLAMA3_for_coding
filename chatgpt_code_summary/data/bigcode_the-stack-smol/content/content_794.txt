"""inter-base steganography
producing base32 and base64 decodable strings"""
from base64 import b64encode, b64decode
import string
from itertools import product
from argparse import ArgumentParser

CHARSET = string.printable.encode()
B32_CHARSET = (string.ascii_uppercase + '234567').encode()
B64_CHARSET = (
    string.ascii_lowercase +
    string.ascii_uppercase +
    string.digits +
    '+/').encode()
ASCII_LOWER = string.ascii_lowercase.encode()
WHITESPACE = string.whitespace.encode()
ALPHA_SPACE = (
    string.ascii_uppercase +
    string.ascii_lowercase +
    string.whitespace).encode()

ASCII_SUBS = {"a": ["a", "A", "4", "@"],
              "b": ["b", "B", "8", "6"],
              "c": ["c", "C", "("],
              "d": ["d", "D"],
              "e": ["e", "E", "3"],
              "f": ["f", "F"],
              "g": ["g", "G", "6", "9"],
              "h": ["h", "H", "#"],
              "i": ["i", "I", "1", "|", "!"],
              "j": ["j", "J", "]", ";"],
              "k": ["k", "K"],
              "l": ["l", "L", "1", "|"],
              "m": ["m", "M"],
              "n": ["n", "N"],
              "o": ["o", "O", "0"],
              "p": ["p", "P"],
              "q": ["q", "Q", "9"],
              "r": ["r", "R", "2"],
              "s": ["s", "S", "5", "$"],
              "t": ["t", "T", "7", "+"],
              "u": ["u", "U"],
              "v": ["v", "V"],
              "w": ["w", "W"],
              "x": ["x", "X"],
              "y": ["y", "Y"],
              "z": ["z", "Z", "2", "%"],
              "0": ["0"],
              "1": ["1"],
              "2": ["2"],
              "3": ["3"],
              "4": ["4"],
              "5": ["5"],
              "6": ["6"],
              "7": ["7"],
              "8": ["8"],
              "9": ["9"],
              " ": [" ", "\t", "_"]
              }


def all_variations(word: str) -> list:
    """
    Produce all single-character leet variations of a string
    """
    ans = [""]
    for leet_letter in [ASCII_SUBS[i] for i in word]:
        ans = [x + y for x in ans for y in leet_letter]
    return ans


def variation_gen(word: str):
    """
    Produces all single-character leet variations of a string

    Args:
        word: a 3 character string to generate all variations

    Returns:
        generator: generator for all possible leet variations
    """
    return product(*(ASCII_SUBS[i] for i in word))


def all_valid_variations(word: str) -> list:
    """
    Returns all leet variations of a triplet which result in a
    Base32 only charset words on base64 encoding

    Args:
        word: An english triplet
    Returns:
        list: of all valid variations
    """
    result = []
    for variation in variation_gen(word):
        if all(i in B32_CHARSET for i in b64encode(
                ''.join(variation).encode())):
            result.append("".join(variation))
    return result


def valid_variation(word: str) -> str:
    """
    Generates a single valid variation

    Args:
        word: the triplet to generate a variation from
    Returns:
        str: A valid variation of `word` or None otherwise
    """
    for variation in variation_gen(word):
        if all(i in B32_CHARSET for i in b64encode(
                ''.join(variation).encode())):
            return "".join(variation)
    return None


# List to precompute the triplets for which there doesnt exist a valid
# variation
NON_LEET = []
for perm in product(string.ascii_lowercase + ' ' + string.digits, repeat=3):
    if not valid_variation(''.join(perm)):
        NON_LEET.append(''.join(perm))


def transform(strng: str) -> str:
    """
    Transform the string to only lower alpha and numerics and spaces
    Converts uppercase to lower case and strips all other characters except
    space
    """
    for char in string.punctuation + string.whitespace[1:]:
        strng = strng.replace(char, '')
    return strng.lower() + ' ' * (8 - len(strng) % 8)


def master_encode(strng: str) -> bytes:
    """
    Encodes a string to its leet equivalent (sans punctuation) which when
    base64 encoded contains only base32 characters
    """
    if isinstance(strng, (bytes, bytearray)):
        strng = strng.decode()
    strng = transform(strng)
    result = ''
    i = 0
    while i < len(strng):
        try:
            current = strng[i:i + 3]
            if current in NON_LEET:
                if current[:2] + ' ' not in NON_LEET:
                    result += valid_variation(current[:2] + ' ')
                    i += 2
                elif current[0] + '  ' not in NON_LEET:
                    result += valid_variation(current[0] + '  ')
                    i += 1
                elif ' {} '.format(current[0]) not in NON_LEET:
                    result += valid_variation(' {} '.format(current[0]))
                    i += 1
                elif '  {}'.format(current[0]) not in NON_LEET:
                    result += valid_variation('  {}'.format(current[0]))
                    i += 1
                else:
                    i += 1
            else:
                result += valid_variation(current)
                i += 3
        except TypeError:
            i += 1
    return b64encode(result.encode())


if __name__ == "__main__":
    PARSER = ArgumentParser(description="")
    PARSER.add_argument(
        '--input',
        help='read a single line directly from input',
        action="store_true")
    PARSER.add_argument(
        '--show',
        help='shows the transformed input which results in correct encoding',
        action="store_true")
    PARSER.add_argument(
        '--file',
        help='reading text from file for conversion',
        action="append")
    ARGS = PARSER.parse_args()
    TEST_STRING = """Steganography  is the practice of concealing a file,
     message, image, or video within another file, message, image, or video.
    The word steganography comes from Greek steganographia, which combines
    the words steganos meaning "covered or concealed", and graphia meaning
    "writing". The first recorded use of the term was by Johannes Trithemius
    in his Steganographia, a treatise on cryptography and steganography,
    disguised as a book on magic. Generally, the hidden messages appear to
    be (or to be part of) something else: images, articles, shopping lists,
    or some other cover text. For example, the hidden message may be in
    invisible ink between the visible lines of a private letter. Some
    implementations of steganography that lack a shared secret are forms
    of security through obscurity, and key-dependent steganographic schemes
    adhere to Kerckhoffs's principle."""
    if ARGS.file:
        with open(ARGS.file[0], 'rb') as inp_file:
            TEST_STRING = inp_file.read()
    else:
        TEST_STRING = input("input the line to encode:\n")
    ENCODED_STRING = master_encode(TEST_STRING)
    print("ENCODED STRING: {}".format(ENCODED_STRING))
    if ARGS.show:
        print("Transformed string: {}".format(b64decode(ENCODED_STRING)))
    # WTBVICAJV2VSZSBFWHBFY3RJIG4JOSBGTGFHNSBCVXQJYTFMICAJWTBVIDZFVCBJNSB3ZTFS\
    # ZCBCYXNFNSBCYSAJTWJPMDJMZSAJTWVOVCBET25UICAJICB3T3JSWSBJVHMJIGYJVW4JIG4JZXZ\
    # FIHIJVCNFTGVTNSAJ
