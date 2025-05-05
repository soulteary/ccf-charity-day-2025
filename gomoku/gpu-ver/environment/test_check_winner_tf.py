import unittest
import tensorflow as tf
from .utils import check_winner_tf

class TestCheckWinnerTF(unittest.TestCase):

    def test_horizontal_win(self):
        boards = tf.zeros((1, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, 7, i] for i in range(5)], [1]*5)
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 1)

    def test_vertical_win(self):
        boards = tf.zeros((1, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, i, 10] for i in range(5)], [2]*5)
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 2)

    def test_diagonal_win_main(self):
        boards = tf.zeros((1, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, i, i] for i in range(5)], [1]*5)
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 1)

    def test_diagonal_win_anti(self):
        boards = tf.zeros((1, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, i, 14 - i] for i in range(5)], [2]*5)
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 2)

    def test_no_winner(self):
        boards = tf.zeros((1, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], [1, 1, 1, 2])
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 0)

    def test_batch_multiple_winners(self):
        boards = tf.zeros((2, 15, 15), dtype=tf.int32)
        boards = tf.tensor_scatter_nd_update(boards, [[0, 3, i] for i in range(5)], [1]*5)
        boards = tf.tensor_scatter_nd_update(boards, [[1, i, 4] for i in range(5)], [2]*5)
        winner = check_winner_tf(boards).numpy()
        self.assertEqual(winner[0], 1)
        self.assertEqual(winner[1], 2)

if __name__ == '__main__':
    unittest.main()
