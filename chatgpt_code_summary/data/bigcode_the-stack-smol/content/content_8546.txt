from unittest import TestCase, mock

from bulksms.sms import send_single, send_bulk


class BulkSMSTestCase(TestCase):
    def test_send_single_sms(self):
        # Mock send single sms function.
        mock_send_single = mock.create_autospec(send_single, return_value='results')
        mock_send_single('0831234567', 'Message.')
        mock_send_single.assert_called_once_with('0831234567', 'Message.')
        self.assertEqual(mock_send_single.call_count, 1)

    def test_send_bulk_sms(self):
        # Mock send bulk sms function.
        mock_send_bulk = mock.create_autospec(send_bulk, return_value='results')
        mock_send_bulk('test.txt')
        mock_send_bulk.assert_called_once_with('test.txt')
        self.assertEqual(mock_send_bulk.call_count, 1)
