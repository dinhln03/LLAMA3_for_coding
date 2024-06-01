import unittest
import io
import sys
import random
from unittest.mock import MagicMock, Mock, patch
from snap.grid import Grid
from snap.hand import Hand
from snap.card import Card


class TestGrid(unittest.TestCase):
    def test__get__origin__returns_correct_cards(self):
        random.seed(1)
        expected_card = Card(7)
        grid = Grid(3)
        mock_position = self.get_mock_position(2, 1)
        self.assertEqual(expected_card, grid.get(mock_position))

    @patch.object(Hand, "hide_all")
    def test__hide_all__calls_hide_all_on_hand(self, mock_hide_all):
        height = 3
        grid = Grid(height)
        grid.hide_all()
        mock_hide_all.assert_called()
        self.assertEqual(height, len(mock_hide_all.call_args_list))

    @patch.object(Hand, "strings")
    def test__strings__returns_mock_strings(self, mock_strings_method):
        mock_strings = ["line 1", "line 2"]
        mock_strings_method.return_value = mock_strings
        height = 3
        grid = Grid(height)
        strings = grid.strings()
        mock_strings_method.assert_called()
        self.assertEqual(height, len(mock_strings_method.call_args_list))
        self.assertEqual(mock_strings * height, strings)

    def get_mock_position(self, x, y):
        pos = Mock()
        pos.x.return_value = x
        pos.y.return_value = y
        return pos

    def get_mock_hand(self):
        hand = Mock()
        hand.hide_all = MagicMock()
        return hand