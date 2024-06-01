import sys
from os import listdir
from os.path import isfile, join, dirname, realpath
import struct
import gzip


def list_dir(d):
    return [f for f in listdir(d) if isfile(join(d, f))]


def store(p, file):
    try:
        output_file = open(p, "w", encoding="utf-8", errors="xmlcharrefreplace")
        output_file.write(file)
    except:
        print("Unable to store the file. Error:", sys.exc_info()[0])
        raise


def store_bin(p, file):
    with open(p, 'wb') as f:
        if isinstance(file, int):
            f.write(struct.pack('i', file))  # write an int
        elif isinstance(file, str):
            f.write(file)  # write a string
        else:
            raise TypeError('Can only write str or int')


def load(p, compression=None):
    if compression == 'gz' or compression == 'gzip':
        f = gzip.open(p, 'rb')
    else:
        f = open(p, mode="r", encoding="utf-8")

    content = f.read()
    f.close()
    return content


def to_string(data: bytes, encoding=None):
    if encoding is None:
        return data.decode("utf-8")
    return data.decode(encoding)


def store_list(p, files, file_names):
    for i in len(files):
        store(p + file_names[i], files[i])
        i += 1


def path(file):
    return dirname(realpath(file)).replace("\\", "/")
