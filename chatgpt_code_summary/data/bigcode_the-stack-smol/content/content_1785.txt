from typing import List
import unittest
from model import tasks


class TestRepository(unittest.TestCase):
    def test_list(self):
        rep = tasks.Repository()

        l = rep.list()
        self.assertEqual(len(l), 2)
        self.assertEqual(l[0].id, 1)
        self.assertEqual(l[0].text, "task1")
        self.assertEqual(l[0].done, False)
        self.assertEqual(l[1].id, 2)

        rep._tasks[0].done = True
        l = rep.list()
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].id, 2)
        self.assertEqual(l[0].done, False)

    def test_add(self):
        rep = tasks.Repository()

        task = tasks.Task(100, "new task")
        rep.add(task)

        l = rep.list()

        self.assertEqual(len(l), 3)
        self.assertEqual(l[2].id, 3)
        self.assertEqual(l[2].text, "new task")
        self.assertEqual(l[2].done, False)

    def test_done(self):
        rep = tasks.Repository()

        rep.done(1)

        l = rep.list()
        self.assertEqual(len(l), 1)
        self.assertEqual(l[0].id, 2)
        self.assertEqual(l[0].done, False)
