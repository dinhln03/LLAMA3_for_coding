#!/usr/bin/python3
# -*- coding: utf-8 -*-
#pylint: skip-file

from nose.tools import assert_equal
from iot_message.cryptor.plain import Cryptor
from iot_message.message import Message

__author__ = 'Bartosz Kościów'

import iot_message.factory as factory


class TestCryptorPlain(object):

    def setUp(self):
        Message.chip_id = 'pc'
        Message.node_name = 'Turkusik'
        Message.drop_unencrypted = False
        Message.encoders = []
        Message.decoders = {}

    def test_encode_message(self):
        Message.add_encoder(Cryptor())
        msg = factory.MessageFactory.create()
        inp = {"event": "channel.on", "parameters": {"channel": 0}, "response": "", "targets": ["node-north"]}
        msg.set(inp)
        msg.encrypt()

        assert_equal(inp["event"], msg.data["event"])
        assert_equal(inp["parameters"], msg.data["parameters"])
        assert_equal(inp["targets"], msg.data["targets"])

    def test_decrypt_message(self):
        Message.add_decoder(Cryptor())
        inp = """{"protocol": "iot:1", "node": "Turkusik", "chip_id": "pc", "event": "message.plain", "parameters": ["a"], "response": "", "targets": ["Turkusik"]}"""
        msg = factory.MessageFactory.create(inp)
        assert_equal(msg.data["event"], "message.plain")
        assert_equal(msg.data["parameters"], ["a"])
        assert_equal(msg.data["targets"], ['Turkusik'])
