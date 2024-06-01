from unittest import TestCase
from btcmagic import transaction, convert
import os
import json

class TestTransaction(TestCase):

    def setUp(self):
        self.tx_bin = convert.hex_to_bytes(
            '0100000001637aaf20d708fcff67bb688af6e41d1807e6883f736c50eacb6042bf6e6c829c010000008c493046022100da1e59d78bb88ca7c3e13a4a6f4e259d5dd8cb177d5f79199bf024b1f57121d50221008d1d9838606a62ed4bd011a6ce8a2042ae2dc38fd05381b50aa388a1c8bd9150014104d3b615c609e48ae81389f6617b50473bf4c93f63c9853cd038aa4f00a989ebd62ae8253555e24c88b939817da18cd4e7263fda6a0e815097589bb90a5a6b3ff1ffffffff03b9000000000000001976a9149fe14d50c95abd6ecddc5d61255cfe5aebeba7e988ac57300f00000000001976a914c0492db5f283a22274ef378cdffbe5ecbe29862b88ac00000000000000000a6a0810e2cdc1af05180100000000')

        self.tx_obj = {
            'ins': [
                {
                    'sequence': 4294967295,
                    'script': b'I0F\x02!\x00\xda\x1eY\xd7\x8b\xb8\x8c\xa7\xc3\xe1:JoN%\x9d]\xd8\xcb\x17}_y\x19\x9b\xf0$\xb1\xf5q!\xd5\x02!\x00\x8d\x1d\x988`jb\xedK\xd0\x11\xa6\xce\x8a B\xae-\xc3\x8f\xd0S\x81\xb5\n\xa3\x88\xa1\xc8\xbd\x91P\x01A\x04\xd3\xb6\x15\xc6\t\xe4\x8a\xe8\x13\x89\xf6a{PG;\xf4\xc9?c\xc9\x85<\xd08\xaaO\x00\xa9\x89\xeb\xd6*\xe8%5U\xe2L\x88\xb99\x81}\xa1\x8c\xd4\xe7&?\xdaj\x0e\x81P\x97X\x9b\xb9\nZk?\xf1',
                    'outpoint': {'index': 1, 'hash': b'\x9c\x82ln\xbfB`\xcb\xeaPls?\x88\xe6\x07\x18\x1d\xe4\xf6\x8ah\xbbg\xff\xfc\x08\xd7 \xafzc'}
                }
            ],
            'locktime': 0,
            'version': 1,
            'outs': [
                {
                    'value': 185,
                    'script': b'v\xa9\x14\x9f\xe1MP\xc9Z\xbdn\xcd\xdc]a%\\\xfeZ\xeb\xeb\xa7\xe9\x88\xac'
                },
                {
                    'value': 995415,
                    'script': b'v\xa9\x14\xc0I-\xb5\xf2\x83\xa2"t\xef7\x8c\xdf\xfb\xe5\xec\xbe)\x86+\x88\xac'
                },
                {
                    'value': 0,
                    'script': b'j\x08\x10\xe2\xcd\xc1\xaf\x05\x18\x01'
                }
            ]
        }

    def test_deserialization(self):
        tx_obj = transaction.deserialize(self.tx_bin)
        self.assertEqual(tx_obj, self.tx_obj)

    def test_serialization(self):
        tx_bin = transaction.serialize(self.tx_obj)
        self.assertEqual(tx_bin, self.tx_bin)


class TestSighash(TestCase):
    def setUp(self):
        loc = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(loc, 'sighash.json')) as f:
            self.data = json.load(f)

    def test_sighash(self):
        first = True
        for vector in self.data:

            # Ignore first header row in the JSON.
            if first:
                first = False
                continue

            tx = transaction.deserialize(convert.hex_to_bytes(vector[0]))
            script = convert.hex_to_bytes(vector[1])
            index = int(vector[2])
            hashtype = int(vector[3]) & 0xffffffff  # This must be unsigned int
            sighash = convert.hex_to_bytes(vector[4])[::-1]  # It's reversed for some reason?

            my_sighash = transaction.sighash(tx, index, script, hashtype)

            self.assertEqual(
                sighash,
                my_sighash,
                'hashtype = {:x}'.format(hashtype)
            )
