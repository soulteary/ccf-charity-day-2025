import tensorflow as tf
from .utils import check_winner_tf

class GomokuGPUEnvironment:
    def __init__(self, batch_size, board_size=15):
        self.batch_size = batch_size
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.boards = tf.Variable(tf.zeros((self.batch_size, self.board_size, self.board_size), dtype=tf.int32))
        self.current_players = tf.Variable(tf.ones((self.batch_size,), dtype=tf.int32))
        self.done = tf.Variable(tf.zeros((self.batch_size,), dtype=tf.bool))
        self.winners = tf.Variable(tf.zeros((self.batch_size,), dtype=tf.int32))

    def step(self, actions):
        indices = tf.stack([
            tf.range(self.batch_size, dtype=tf.int32),
            actions[:, 0],
            actions[:, 1]
        ], axis=-1)
        updates = tf.where(~self.done, self.current_players, 0)
        self.boards.assign(tf.tensor_scatter_nd_update(self.boards, indices, updates))
        done, winners = self._check_win_conditions()
        self.done.assign(done)
        self.winners.assign(winners)
        self.current_players.assign(tf.where(~self.done, 3 - self.current_players, self.current_players))

    @tf.function
    def _check_win_conditions(self):
        winner = check_winner_tf(self.boards)
        return winner > 0, winner
