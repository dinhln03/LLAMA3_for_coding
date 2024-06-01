from collections import namedtuple
import json
import os
import unittest

import context
import ansi
import comment

class TestComment(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

        comments_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    'comments.json'
                    )
                )
        with open(comments_path) as f:
            self.comments_data = json.load(f)

        file_contents_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    'file_contents.txt'
                    )
                )
        with open(file_contents_path) as f:
            text = f.read()
            self.file_contents = text.split('\n')

    def test_create_comment_no_context(self):
        filename = 'file2'
        data = self.comments_data[filename][0]
        c = comment.Comment(data, filename)
        self.assertEqual('4', c.id)
        self.assertEqual('3', c.patch_set)
        self.assertEqual(None, c.parent)
        self.assertEqual('Name1', c.author)
        self.assertEqual('A file comment', c.message)
        self.assertEqual('2021-04-24', c.date)
        self.assertEqual(filename, c.file)
        self.assertEqual('', c.context[0])
        self.assertEqual('', c.context[1])
        self.assertEqual('', c.context[2])

    def test_create_comment_line(self):
        filename = 'file1'
        data = self.comments_data[filename][2]
        c = comment.Comment(data, filename, self.file_contents)
        self.assertEqual('', c.context[0])
        self.assertEqual('Some more content.', c.context[1])
        self.assertEqual('', c.context[2])

    def test_create_comment_range_one_line(self):
        filename = 'file2'
        data = self.comments_data[filename][1]
        c = comment.Comment(data, filename, self.file_contents)
        self.assertEqual('File ', c.context[0])
        self.assertEqual('starts', c.context[1])
        self.assertEqual(' here.', c.context[2])

    def test_create_comment_range_four_lines(self):
        filename = 'file1'
        data = self.comments_data[filename][0]
        c = comment.Comment(data, filename, self.file_contents)
        self.assertEqual('File ', c.context[0])
        self.assertEqual('starts here.\nSome content.\nSome more content.\nThis', c.context[1])
        self.assertEqual(' is the end.', c.context[2])

    def test_str(self):
        filename = 'file1'
        data = self.comments_data[filename][0]
        c = comment.Comment(data, filename, self.file_contents)
        actual = str(c)
        expected = ' '.join([
            'Name1',
            ansi.format('Can you update this, Name2?', [ansi.GREEN, ansi.ITALIC])
            ])
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main(verbosity=2)
