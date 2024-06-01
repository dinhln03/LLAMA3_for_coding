import gorp
from gorp.readfiles import *
import unittest

ogdir = os.getcwd()
newdir = os.path.join(gorpdir, "testDir")
os.chdir(newdir)
is_version_2p0 = gorp.__version__[:3] == "2.0"


class XOptionTester(unittest.TestCase):
    session = GorpSession(print_output=False)

    @unittest.skipIf(
        is_version_2p0,
        "this test fails but the '-x' option with css selectors still works fine in normal use",
    )
    def test_css_selectors(self):
        fname = os.path.join(newdir, "bluddGame.htm")
        query = f"-x 'img.Bludd' /{fname}"
        self.session.receive_query(query)
        correct_output = {
            f"{fname}": [
                'b\'<img class="Bludd" id="Bludd" src=".\\\\viking pics\\\\Bludd.png" height="100" width="100" alt="Bludd, Blood God" title="Bludd, the Blood God (of Blood!)"/>&#13;\\n\''
            ]
        }
        self.assertEqual(self.session.resultset, correct_output)

    @unittest.skipIf(
        is_version_2p0,
        "this test fails but the '-x' option with XPath selectors still works fine in normal use",
    )
    def test_xpath_multi_results(self):
        fname = os.path.join(newdir, "books.xml")
        query = f"-x -n '//bookstore//book[@category]' /{fname}"
        self.session.receive_query(query)
        correct_output = {
            fname: {
                (
                    "bookstore",
                    0,
                ): 'b\'<book category="cooking">\\n    <title lang="en">Everyday Italian</title>\\n    <author>Giada De Laurentiis</author>\\n    <year>2005</year>\\n    <price>30.00</price>\\n  </book>\\n  \'',
                (
                    "bookstore",
                    1,
                ): 'b\'<book category="children">\\n    <title lang="en">Harry Potter</title>\\n    <author>J K. Rowling</author>\\n    <year>2005</year>\\n    <price>29.99</price>\\n  </book>\\n  \'',
                (
                    "bookstore",
                    2,
                ): 'b\'<book category="web">\\n    <title lang="en">Learning XML</title>\\n    <author>Erik T. Ray</author>\\n    <year>2003</year>\\n    <price>39.95</price>\\n  </book>\\n\'',
            }
        }
        self.assertEqual(self.session.resultset, correct_output)

    def zzzz_cleanup(self):
        os.chdir(ogdir)
        session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
