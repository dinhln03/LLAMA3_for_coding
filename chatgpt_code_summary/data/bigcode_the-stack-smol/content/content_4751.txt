import unittest

from my_lambdata.assignment1 import WrangledFrame


class TestWrangledFrame(unittest.TestCase):

    def test_add_state_names(self):
        wf = WrangledFrame({"abbrev": ["CA", "CO", "CT", "DC", "TX"]})

        breakpoint()
        wf.add_state_names()
        # ensure there is a "name" column
        self.assertEqual(list(wf.columns), ['abbrev', 'name'])
        # ensure the values of WF are specific classes/values
        # (string, "California")
        self.assertEqual(wf["name"][0], "California")
        self.assertEqual(wf["abbrev"][0], "CA")


if __name__ == '__main__':
    unittest.main()
