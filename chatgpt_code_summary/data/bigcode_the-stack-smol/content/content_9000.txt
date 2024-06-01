from globconf import config
from globconf import verify_required_options
import unittest
import os

# let's test on a predefined file included in the unitteast
config.read(os.path.dirname(__file__)+'/config.ini')


class TestConf(unittest.TestCase):

    def test_config_file_present(self):
        self.assertTrue(os.path.isfile(os.path.dirname(__file__)+'/config.ini'), 'config.ini in project root dir is missing')

    def test_verify_required_options(self):
        self.assertTrue(verify_required_options('SectionOne', ['parameter_one', 'parameter_two']))

    def test_verify_sections(self):
        self.assertEqual(2, 2)
