import unittest
import numpy as np
import tensorflow as tf
from agents.gpu_mcts import GPUMCTS

class TestGPUMCTS(unittest.TestCase):

    def setUp(self):
        self.mcts = GPUMCTS(iterations=100)

    def test_search_basic_functionality(self):
        boards = np.zeros((1, 15, 15), dtype=int)
        players = np.array([1], dtype=int)

        moves = self.mcts.search(
            tf.constant(boards, dtype=tf.int32),
            tf.constant(players, dtype=tf.int32)
        ).numpy()

        self.assertEqual(moves.shape, (1, 2))
        self.assertTrue((moves >= 0).all() and (moves < 15).all())

    def test_search_on_almost_full_board(self):
        boards = np.ones((1, 15, 15), dtype=int)
        boards[0, 7, 7] = 0
        players = np.array([2], dtype=int)

        moves = self.mcts.search(
            tf.constant(boards, dtype=tf.int32),
            tf.constant(players, dtype=tf.int32)
        ).numpy()

        self.assertEqual(tuple(moves[0]), (7, 7))

    def test_sample_moves_no_empty_space(self):
        board = np.ones((1, 15, 15), dtype=int)
        moves = self.mcts.sample_moves(tf.constant(board, dtype=tf.int32)).numpy()

        # When no empty space exists, return should be default (0, 0)
        self.assertEqual(moves.shape, (1, 2))
        self.assertTrue(np.array_equal(moves[0], [0, 0]))

    def test_sample_moves_with_empty_space(self):
        board = np.zeros((1, 15, 15), dtype=int)
        board[0, 7, 7] = 1
        moves = self.mcts.sample_moves(tf.constant(board, dtype=tf.int32)).numpy()

        self.assertEqual(moves.shape, (1, 2))
        self.assertFalse(np.array_equal(moves[0], [7, 7]))

if __name__ == '__main__':
    tf.random.set_seed(42)
    unittest.main()
