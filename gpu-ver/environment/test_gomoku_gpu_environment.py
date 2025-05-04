import unittest
import tensorflow as tf
from .gomoku_gpu_env import GomokuGPUEnvironment

class TestGomokuGPUEnvironment(unittest.TestCase):

    def test_initialization(self):
        env = GomokuGPUEnvironment(batch_size=2, board_size=15)
        self.assertEqual(env.boards.shape, (2, 15, 15))
        self.assertTrue(tf.reduce_all(env.boards == 0))
        self.assertTrue(tf.reduce_all(env.current_players == 1))
        self.assertTrue(tf.reduce_all(~env.done))
        self.assertTrue(tf.reduce_all(env.winners == 0))

    def test_step_function(self):
        env = GomokuGPUEnvironment(batch_size=1, board_size=15)
        env.reset()

        action = tf.constant([[7, 7]], dtype=tf.int32)
        env.step(action)

        board_np = env.boards.numpy()
        self.assertEqual(board_np[0, 7, 7], 1)
        self.assertEqual(env.current_players.numpy()[0], 2)
        self.assertFalse(env.done.numpy()[0])

    def test_win_detection_horizontal(self):
        env = GomokuGPUEnvironment(batch_size=1, board_size=15)
        env.reset()

        moves = [[7, i] for i in range(5)]
        for move in moves:
            action = tf.constant([move], dtype=tf.int32)
            env.step(action)
            env.current_players.assign(tf.ones_like(env.current_players))  # 强制当前玩家为1

        self.assertTrue(env.done.numpy()[0])
        self.assertEqual(env.winners.numpy()[0], 1)

    def test_win_detection_vertical(self):
        env = GomokuGPUEnvironment(batch_size=1, board_size=15)
        env.reset()

        moves = [[i, 8] for i in range(5)]
        for move in moves:
            action = tf.constant([move], dtype=tf.int32)
            env.step(action)
            env.current_players.assign(tf.ones_like(env.current_players))  # 强制当前玩家为1

        self.assertTrue(env.done.numpy()[0])
        self.assertEqual(env.winners.numpy()[0], 1)

    def test_win_detection_diagonal(self):
        env = GomokuGPUEnvironment(batch_size=1, board_size=15)
        env.reset()

        moves = [[i, i] for i in range(5)]
        for move in moves:
            action = tf.constant([move], dtype=tf.int32)
            env.step(action)
            env.current_players.assign(tf.ones_like(env.current_players))  # 强制当前玩家为1

        self.assertTrue(env.done.numpy()[0])
        self.assertEqual(env.winners.numpy()[0], 1)

    def test_no_win_condition(self):
        env = GomokuGPUEnvironment(batch_size=1, board_size=15)
        env.reset()

        moves = [[0, 0], [0, 1], [0, 2], [1, 0]]
        for move in moves:
            action = tf.constant([move], dtype=tf.int32)
            env.step(action)

        self.assertFalse(env.done.numpy()[0])
        self.assertEqual(env.winners.numpy()[0], 0)
        self.assertEqual(env.current_players.numpy()[0], 1)

if __name__ == '__main__':
    unittest.main()
