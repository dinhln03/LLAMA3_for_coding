#!/usr/bin/env python
#
# @file test_signals.py
#
# @author Matt Gigli <mjgigli@gmail.com>
#
# @section LICENSE
#
# The MIT License (MIT)
# Copyright (c) 2016 Matt Gigli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
#

import unittest
from ao.signals import dispatcher


class test_signals(unittest.TestCase):
    def setUp(self):
        self.cb1 = 0
        self.cb2 = 0
        self.cb3 = 0
        self.cb4 = 0
        self.cb5 = 0
        self.cb6 = 0
        self.cb_arg1 = None
        self.cb_arg2 = None

    def tearDown(self):
        dispatcher.unsubscribe_all()

    def callback_1(self):
        self.cb1 = 1

    def callback_2(self):
        self.cb2 = 2

    def callback_1234(self):
        self.cb1 = 1
        self.cb2 = 2
        self.cb3 = 3
        self.cb4 = 4

    def callback_34(self):
        self.cb3 = 3
        self.cb4 = 4

    def callback_56(self):
        self.cb5 = 5
        self.cb6 = 6

    def callback_args(self, arg1, arg2=None):
        self.cb_arg1 = arg1
        self.cb_arg2 = arg2

    def test_one_receiver(self):
        # verify test class members are reset
        self.assertEqual(self.cb1, 0)

        # subscribe to the cb1_sig signal
        dispatcher.subscribe('cb1_sig', self.callback_1)

        # publish the signal to call the callback
        dispatcher.publish('cb1_sig')

        # verify callback was called
        self.assertEqual(self.cb1, 1)

    def test_arguments(self):
        # verify test class members are reset
        self.assertEqual(self.cb_arg1, None)
        self.assertEqual(self.cb_arg2, None)

        # subscribe to the cbarg_sig signal
        dispatcher.subscribe('cbarg_sig', self.callback_args)

        # publish the signal to call the callback
        dispatcher.publish('cbarg_sig', 'arg1', arg2='arg2')

        # verify callback was called
        self.assertEqual(self.cb_arg1, 'arg1')
        self.assertEqual(self.cb_arg2, 'arg2')

    def test_multiple_receivers(self):
        # verify test class members are reset
        self.assertEqual(self.cb1, 0)
        self.assertEqual(self.cb2, 0)
        self.assertEqual(self.cb3, 0)
        self.assertEqual(self.cb4, 0)

        # subscribe to the signals
        dispatcher.subscribe('cb2_sig', self.callback_1234)

        # publish the signal to call the callbacks
        dispatcher.publish('cb2_sig')

        # verify callbacks were called
        self.assertEqual(self.cb1, 1)
        self.assertEqual(self.cb2, 2)
        self.assertEqual(self.cb3, 3)
        self.assertEqual(self.cb4, 4)

    def test_publish_unsubscribed_signal(self):
        # publish a signal that hasn't been subscribed to, to verify that no
        # error occurs when publishing such a signal
        dispatcher.publish('lonely_sig')

    def test_unsubscribe_unsubscribed_signal(self):
        # verify no exception is raised when unsubscribing a receiver from a
        # signal that was never subscribed to
        dispatcher.unsubscribe('lonely_sig', self.callback_1)

    def test_unsubscribe(self):
        # subscribe, publish and check that callback was called
        dispatcher.subscribe('cb1_sig', self.callback_1)
        dispatcher.publish('cb1_sig')
        self.assertEqual(self.cb1, 1)

        # reset cb1, unsubscribe and show that callback is not called
        self.cb1 = 0
        dispatcher.unsubscribe('cb1_sig', self.callback_1)
        dispatcher.publish('cb1_sig')
        self.assertEqual(self.cb1, 0)

    def test_unsubscribe_all_for_signal(self):
        # subscribe some receivers for some signals
        dispatcher.subscribe('cb1_sig', self.callback_1)
        dispatcher.subscribe('cb1_sig', self.callback_2)
        dispatcher.subscribe('cb3_sig', self.callback_34)
        dispatcher.subscribe('cb3_sig', self.callback_56)

        # unsuscribe just for cb1_sig
        dispatcher.unsubscribe_all('cb1_sig')

        # verify only cb1_sig receivers were unsubscribed
        dispatcher.publish('cb1_sig')
        dispatcher.publish('cb3_sig')
        self.assertEqual(self.cb1, 0)
        self.assertEqual(self.cb2, 0)
        self.assertEqual(self.cb3, 3)
        self.assertEqual(self.cb4, 4)
        self.assertEqual(self.cb5, 5)
        self.assertEqual(self.cb6, 6)

    def test_unsubscribe_all(self):
        dispatcher.subscribe('cb1_sig', self.callback_1)
        dispatcher.subscribe('cb1_sig', self.callback_2)
        dispatcher.subscribe('cb3_sig', self.callback_34)
        dispatcher.subscribe('cb3_sig', self.callback_56)

        # unsuscribe all signals
        dispatcher.unsubscribe_all()

        # verify all receivers were unsubscribed
        dispatcher.publish('cb1_sig')
        dispatcher.publish('cb3_sig')
        self.assertEqual(self.cb1, 0)
        self.assertEqual(self.cb2, 0)
        self.assertEqual(self.cb3, 0)
        self.assertEqual(self.cb4, 0)
        self.assertEqual(self.cb5, 0)
        self.assertEqual(self.cb6, 0)
