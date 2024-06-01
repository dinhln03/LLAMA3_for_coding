from models import Supervisor
import unittest


class SupervisorTestCase(unittest.TestCase):

    def setUp(self):
        self.supervisor = Supervisor.login('Mohammad', '1234', '0123456')
        self.sample = Supervisor.sample()

    def test_all_data(self):
        self.assertIsInstance(self.supervisor, Supervisor,
                              "Sample does not return proper instance")
        self.assertTrue(hasattr(self.supervisor, 'username'),
                        "Instance does not have username")
        self.assertTrue(hasattr(self.supervisor, 'password'),
                        "Instance does not have password")
        self.assertTrue(hasattr(self.supervisor, 'phone_number'),
                        "Instance does not have phone_number")
        self.assertFalse(self.sample.logged_in,
                         "Login is not false by default")

    def test_supervisor_protected_method(self):
        self.assertIsNone(self.sample.protected(),
                          "Not raised on protected method")
        self.assertListEqual(self.supervisor.protected(), [1, 2, 3],
                             "Protected data do not match")
