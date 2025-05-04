import unittest
import tensorflow as tf
import numpy as np
from agents.random_simulation import batch_random_simulation
from environment.utils import check_winner_tf

class TestBatchRandomSimulation(unittest.TestCase):

    def generate_strict_draw_board(self):
        board = np.indices((15, 15)).sum(axis=0) % 2 + 1

        # 明确打断所有可能的五连子（横、竖、对角线）
        # 横向打断
        for i in range(15):
            for j in range(11):
                if np.all(board[i, j:j+5] == board[i, j]):
                    board[i, j+2] = 3 - board[i, j]

        # 纵向打断
        for j in range(15):
            for i in range(11):
                if np.all(board[i:i+5, j] == board[i, j]):
                    board[i+2, j] = 3 - board[i, j]

        # 主对角线打断
        for i in range(11):
            for j in range(11):
                if np.all([board[i+k, j+k] == board[i, j] for k in range(5)]):
                    board[i+2, j+2] = 3 - board[i, j]

        # 反对角线打断
        for i in range(4, 15):
            for j in range(11):
                if np.all([board[i-k, j+k] == board[i, j] for k in range(5)]):
                    board[i-2, j+2] = 3 - board[i, j]

        return board

    def test_already_won_board(self):
        board = np.zeros((1, 15, 15), dtype=int)
        board[0, 7, 3:8] = 1
        players = np.array([1])

        winner = batch_random_simulation(tf.constant(board), tf.constant(players), 15).numpy()
        self.assertEqual(winner[0], 1)

    def test_full_board_draw(self):
        board = np.expand_dims(self.generate_strict_draw_board(), axis=0)
        players = np.array([1])

        winner = batch_random_simulation(tf.constant(board), tf.constant(players), 15).numpy()
        self.assertEqual(winner[0], 0)

    def test_one_move_to_win(self):
        board = np.ones((1, 15, 15), dtype=int)
        board[0, 7, 3:7] = 2
        board[0, 7, 7] = 0
        players = np.array([2])

        winner = batch_random_simulation(tf.constant(board), tf.constant(players), 15).numpy()
        self.assertEqual(winner[0], 2)

    def test_random_play_validity(self):
        board = np.zeros((1, 15, 15), dtype=int)
        board[0, 0, 0] = 1
        players = np.array([2])

        winner = batch_random_simulation(tf.constant(board), tf.constant(players), 15).numpy()
        self.assertIn(winner[0], [0, 1, 2])

    def test_batch_multiple_boards(self):
        boards = np.zeros((3, 15, 15), dtype=int)
        players = np.array([1, 2, 1])

        # 已有明确赢家的棋盘
        boards[0, 5, 3:8] = 1

        # 严格无赢家满棋盘
        boards[1] = self.generate_strict_draw_board()

        # 随机局势棋盘
        boards[2, 7, 7] = 1
        boards[2, 7, 8] = 2

        winner = batch_random_simulation(tf.constant(boards), tf.constant(players), 15).numpy()

        self.assertEqual(winner[0], 1)
        self.assertEqual(winner[1], 0)
        self.assertIn(winner[2], [0, 1, 2])

if __name__ == '__main__':
    tf.random.set_seed(42)
    unittest.main()
