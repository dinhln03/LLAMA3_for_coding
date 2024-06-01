# -*- coding: utf-8 -*-

import unittest
from pathlib import Path

from knipse.db import KnipseDB
from knipse.scan import scan_images
from knipse.lists import image_id_from_string
from .test_walk import EXPECTED_IMAGES


class TestKnipseDatabase(unittest.TestCase):

    def setUp(self) -> None:
        self.src = Path(__file__).resolve().parent / 'images' / 'various'
        self.db = KnipseDB(':memory:')

    def test_getting_image_id(self) -> None:
        cnt = 0
        for file_path, progress in scan_images(self.db, self.src,
                                               skip_thumbnail_folders=True):
            cnt += 1
        self.assertEqual(len(EXPECTED_IMAGES), cnt)
        recgn = self.db.get_recognizer()
        image_id = image_id_from_string(str(self.src / 'img_0002.jpg'),
                                        self.src, recgn)
        self.assertEqual(1, image_id)
        image_id = image_id_from_string('I002', self.src, recgn)
        self.assertEqual(2, image_id)
