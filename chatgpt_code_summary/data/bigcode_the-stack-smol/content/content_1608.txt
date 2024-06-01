# Standard Library
import unittest

# YouTubeTimestampRedditBot
from src.utils.youtube import is_youtube_url_without_timestamp


class Youtube(unittest.TestCase):
    def test_is_youtube_url_without_timestamp(self):
        dicts = [
            # no timestamps
            {"input": "https://youtube.com/asdf", "expected_output": True},
            {"input": "wwww.youtube.com/asdf", "expected_output": True},
            {"input": "wwww.youtu.be/asdf", "expected_output": True},
            # has timestamps
            {"input": "https://youtube.com/asdf?t=1m", "expected_output": False},
            {"input": "wwww.youtube.com?watch=asdf&t=1m", "expected_output": False},
            {"input": "wwww.youtu.be/asdf?t=12s", "expected_output": False},
            # not youtube
            {"input": "wwww.asdf.com", "expected_output": False},
            {"input": "https://youfoo.com", "expected_output": False},
        ]

        for (i, d) in enumerate(dicts):
            with self.subTest(i=i):
                assert (
                    is_youtube_url_without_timestamp(d["input"]) == d["expected_output"]
                )
