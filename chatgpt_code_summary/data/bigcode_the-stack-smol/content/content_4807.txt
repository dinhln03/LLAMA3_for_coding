from datetime import datetime
import unittest
from unittest.mock import MagicMock

import numpy as np

from pyhsi.cameras import BaslerCamera


class MockGrab:
    def __init__(self, data):
        self.Array = data

    def GrabSucceeded(self):
        return True

    def Release(self):
        pass


class TestBaslerCamera(unittest.TestCase):
    def setUp(self):
        self.mock_device = MagicMock()
        self.mock_stage = MagicMock()
        self.mock_stage.default_velocity = 20
        self.cam = BaslerCamera(device=self.mock_device)

    def test_capture(self):
        self.mock_device.RetrieveResult = MagicMock(side_effect=[
            MockGrab([[0, 12], [3, 100]]),
            MockGrab([[9, 8], [31, 5]])
        ])
        self.mock_stage.is_moving = MagicMock(side_effect=[True, True, False])
        data = self.cam.capture(self.mock_stage, [0, 100])
        target = np.array([[[12, 100], [0, 3]], [[8, 5], [9, 31]]])
        np.testing.assert_array_equal(data, target)

    def test_file_name_basic(self):
        fn = "test_sample"
        out = self.cam._process_file_name(fn, datetime(2020, 6, 20),
                                          0, 100, 10, (227, 300, 400))
        self.assertEqual(out, "test_sample.hdr")

    def test_file_name_fields(self):
        fn = "sample_{date}_{time}_exp={exp}_{frames}_frames"
        out = self.cam._process_file_name(fn, datetime(2020, 6, 20, 13, 40),
                                          0, 100, 10, (227, 300, 400))
        target = "sample_2020-06-20_13:40:00_exp=4000_227_frames.hdr"
        self.assertEqual(out, target)


if __name__ == "__main__":
    unittest.main()
