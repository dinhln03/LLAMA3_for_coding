# a simple python AES decrypter. Do not remember why I needed this, but nice to have. :)

pw = [255,155,28,115,214,107,206,49,172,65,62,174,19,27,70,79,88,47,108,226,209,225,243,218,126,141,55,107,38,57,78,91]
pw1 = b''
for i in pw:
    pw1 += i.to_bytes(1, 'little')

import sys, hexdump, binascii
from Crypto.Cipher import AES

class AESCipher:
    def __init__(self, key):
        self.key = key

    def decrypt(self, iv, data):
        self.cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self.cipher.decrypt(data)

key = binascii.unhexlify("0602000000a400005253413100040000")
iv = binascii.unhexlify("0100010067244F436E6762F25EA8D704")

raw_un = AESCipher(key).decrypt(iv, pw1)

print(hexdump.hexdump(raw_un))

password = raw_un.decode('utf-16')
print(password)
