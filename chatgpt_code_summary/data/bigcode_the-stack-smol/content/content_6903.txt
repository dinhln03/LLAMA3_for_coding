# -*- coding: utf-8 -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


# imports ######################################################################

import rsa

# participants #################################################################

(sender_pub_key, sender_priv_key) = rsa.newkeys(1024, poolsize=4)

(receiver_pub_key, receiver_priv_key) = rsa.newkeys(1024, poolsize=4)

# side of sender ###############################################################

message = 'Hello'
print('Message:', message)

bin_message = message.encode()
print('Bytes message:', bin_message)

signature = rsa.sign(bin_message, sender_priv_key, 'SHA-512')
print('Signature:', signature)

crypto = rsa.encrypt(bin_message, receiver_pub_key)
print('Encrypted message:', crypto)

# side of receiver

bin_decrypted = rsa.decrypt(crypto, receiver_priv_key)
print('Decrypted bytes message:', bin_decrypted)

if rsa.verify(bin_decrypted, signature, sender_pub_key):
    print('Message signature verified.')
else:
    print('SIGNATURE VERIFICATION ERROR!')

decrypted = bin_decrypted.decode()
print('Decrypted message:', decrypted)

# END ##########################################################################
