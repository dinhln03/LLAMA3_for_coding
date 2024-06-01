# -*- coding: utf-8 -*-
from datetime import datetime
import time
import unittest

from webapp2_caffeine.cache import CacheContainer
from webapp2_caffeine.cache import flush


class DummyCache(CacheContainer):

    key = 'dummy_cache'

    @property
    def fresh_value(self):
        return datetime.now()


class CacheContainerTest(unittest.TestCase):

    def setUp(self):
        flush()

    def tearDown(self):
        flush()

    def test_fresh_value(self):
        container = CacheContainer()
        with self.assertRaises(NotImplementedError):
            container.fresh_value

    def test_set(self):
        container = CacheContainer()
        with self.assertRaises(ValueError):
            container.set('my value')

        container = DummyCache()
        value, expiration = container.set('my value')
        self.assertEqual(value, 'my value')
        self.assertTrue(21000 < expiration - time.time() < 21600)
        self.assertEqual(container.get(), 'my value')

    def test_get(self):
        container = DummyCache()
        self.assertEqual(container.get(), None)
        container.set('my value', 1000)
        self.assertEqual(container.get(), None)
        container.set('my value')
        self.assertEqual(container.get(), 'my value')

    def test_delete(self):
        container = DummyCache()
        container.set('my value')
        container.delete()
        self.assertEqual(container.get(), None)

    def test_update(self):
        container = DummyCache()
        container.update()
        self.assertTrue(container.get())

    def test_value(self):
        container = DummyCache()
        old_value = container.value
        self.assertTrue(old_value)
        self.assertTrue(container.value, old_value)
