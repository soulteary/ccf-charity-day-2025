import unittest
import numpy as np
from .game_utils import check_winner

class TestCheckWinner(unittest.TestCase):

    def test_horizontal_win(self):
        board = np.zeros((15, 15), dtype=int)
        board[7, 3:8] = 1
        self.assertEqual(check_winner(board), 1)

    def test_vertical_win(self):
        board = np.zeros((15, 15), dtype=int)
        board[2:7, 8] = 2
        self.assertEqual(check_winner(board), 2)

    def test_main_diagonal_win(self):
        board = np.zeros((15, 15), dtype=int)
        for i in range(5):
            board[i, i] = 1
        self.assertEqual(check_winner(board), 1)

    def test_anti_diagonal_win(self):
        board = np.zeros((15, 15), dtype=int)
        for i in range(5):
            board[4 - i, i] = 2
        self.assertEqual(check_winner(board), 2)

    def test_draw_condition(self):
        board = np.array([
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
        ])
        self.assertEqual(check_winner(board), -1)

    def test_no_winner_yet(self):
        board = np.zeros((15, 15), dtype=int)
        board[0, :4] = 1
        self.assertEqual(check_winner(board), 0)

if __name__ == '__main__':
    unittest.main()
