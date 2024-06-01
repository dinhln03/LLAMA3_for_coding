import string
import zmq

host = '127.0.0.1'
port = 6789
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind('tcp://%s:%s' % (host, port))

with open('lokomotywa.txt', 'rt') as poem:
    words = poem.read()
for word in words.split():
    word = word.strip(string.punctuation)
    data = word.encode('utf-8')
    if word.startswith(('a','e','i','o','u','y','A','E','I','O','U','Y')):
        pub.send_multipart([b'samogloski', data])
    if len(word) == 5:
        pub.send_multipart([b'piec', data])
