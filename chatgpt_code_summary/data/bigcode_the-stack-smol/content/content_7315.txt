import unittest

from datatypes.exceptions import DataDoesNotMatchSchemaException

from datatypes import postcode_validator


class TestPostcodeValidation(unittest.TestCase):
    def test_can_validate_postcode(self):
        try:
            postcode_validator.validate("WC2B6SE")
            postcode_validator.validate("wc2b6se")
            postcode_validator.validate("wc2b 6se")
            postcode_validator.validate("Wc2b  6se")
        except DataDoesNotMatchSchemaException as e:
            self.fail("Could not validate postcode: " + repr(e))

    def test_does_not_validate_invalid_postcode(self):
        self.assertRaises(DataDoesNotMatchSchemaException, postcode_validator.validate, "sausages")
        self.assertRaises(DataDoesNotMatchSchemaException, postcode_validator.validate, "")
        self.assertRaises(DataDoesNotMatchSchemaException, postcode_validator.validate, 123)

    def test_can_convert_postcode_to_canonical_form(self):
        self.assertEqual(postcode_validator.to_canonical_form("wc2B6sE"), "WC2B 6SE")
        self.assertEqual(postcode_validator.to_canonical_form('pl11aa'), 'PL1 1AA')
        self.assertEqual(postcode_validator.to_canonical_form('pl132aa'), 'PL13 2AA')
        self.assertEqual(postcode_validator.to_canonical_form('pl13 2aa'), 'PL13 2AA')


