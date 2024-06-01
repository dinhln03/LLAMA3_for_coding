import sys
import logging

logger = logging.getLogger(__name__)

import base64
import json
import datetime

from cryptography.hazmat.backends import default_backend

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding

from mqfactory.message.security import Signature

import socket

class Decoded(object):
  def __init__(self, keys):
    self.keys = keys

  def __getitem__(self, name):
    key = self.keys[name]
    if not isinstance(key["public"], rsa.RSAPublicKey):
      return {
        "private": decode(key["private"]),
        "public" : decode(key["public"])
      }
    return key

class RsaSignature(Signature):
  def __init__(self, keys, me=socket.gethostname()):
    self.keys = Decoded(keys)
    self.me   = me
    self.key  = self.keys[self.me]["private"]

  def _sign(self, message, ts=None):
    logger.debug("signing {0}".format(message.id))
    message.tags["signature"] = {
      "origin" : self.me,
      "ts"     : ts or str(datetime.datetime.utcnow())
    }
    payload = serialize(message)
    message.tags["signature"]["hash"] = base64.b64encode(sign(payload, self.key))
  
  def _validate(self, message):
    key = self.keys[message.tags["signature"]["origin"]]["public"]
    signature =  base64.b64decode(message.tags["signature"].pop("hash"))
    payload = serialize(message)    
    validate(payload, signature, key)
    message.tags.pop("signature")

# utility functions wrapping cryptography functions

def generate_key_pair():
  key = rsa.generate_private_key(
     public_exponent=65537,
     key_size=2048,
     backend=default_backend()
  )
  return key, key.public_key()

def encode(key):
  if isinstance(key, rsa.RSAPublicKey):
    return key.public_bytes(
      encoding=serialization.Encoding.PEM,
      format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
  else:
    return key.private_bytes(
      encoding=serialization.Encoding.PEM,
      format=serialization.PrivateFormat.TraditionalOpenSSL,
      encryption_algorithm=serialization.NoEncryption()
    )

def decode(pem):
  try:
    pem = pem.encode("ascii","ignore") # unicode -> str
  except AttributeError:
    pass
  if b"PUBLIC KEY" in pem:
    return serialization.load_pem_public_key(
      pem,
      backend=default_backend()
    )
  else:
    return serialization.load_pem_private_key(
      pem,
      password=None,
      backend=default_backend()
    )

def serialize(message):
  serialized = json.dumps({
    "tags" : message.tags,
    "payload" : message.payload
  }, sort_keys=True).encode("utf-8")
  return base64.b64encode(serialized)

def sign(payload, key):
  return key.sign(
    payload,
    padding.PSS(
      mgf=padding.MGF1(hashes.SHA256()),
      salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
  )
  
def validate(message, signature, key):
  key.verify(
    signature,
    message,
    padding.PSS(
      mgf=padding.MGF1(hashes.SHA256()),
      salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
  )
