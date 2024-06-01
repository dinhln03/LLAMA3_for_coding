from unittest.mock import patch

from django.core.management import call_command
from django.db.utils import OperationalError
from django.test import TestCase


class CommandTests(TestCase):

    def test_wait_for_db_ready(self):
        """Test waiting for db when db is available"""
        with patch('django.db.utils.ConnectionHandler.__getitem__') as gi:
            gi.return_value = True
            call_command('wait_for_db')
            self.assertEqual(gi.call_count, 1)
            # Checking is mock object gi was called only once

    @patch('time.sleep', return_value=True)
    # The patch as a decorator will pass the argument to the test below it
    def test_wait_for_db(self, ts):
        """Test waiting for db"""
        # When the ConnectionHandler raised OperationalError, then it waits for
        # 1 sec and tries again. Delay here can be removed in the unit test by
        # using a patch decorator
        with patch('django.db.utils.ConnectionHandler.__getitem__') as gi:
            gi.side_effect = [OperationalError] * 5 + [True]
            # For 1st 5 tries, it raises OperationalError, then 6th time True
            call_command('wait_for_db')
            self.assertEqual(gi.call_count, 6)
