import tensorflow as tf
import numpy as np
from .random_simulation import batch_random_simulation
from environment.utils import check_winner_tf

class GPUMCTS:
    def __init__(self, iterations, board_size=15):
        self.iterations = iterations
        self.board_size = board_size

    @tf.function
    def search(self, initial_boards, initial_players):
        batch_size = tf.shape(initial_boards)[0]
        initial_mask = tf.where(initial_boards == 0, 0.0, -np.inf)
        move_scores = tf.identity(initial_mask)

        initial_boards = tf.cast(initial_boards, tf.int32)
        initial_players = tf.cast(initial_players, tf.int32)

        for _ in tf.range(self.iterations):
            moves = self.sample_moves(initial_boards)
            sim_boards = tf.identity(initial_boards)
            players = tf.identity(initial_players)

            sim_boards = tf.cast(sim_boards, tf.int32)

            indices = tf.stack([
                tf.range(batch_size, dtype=tf.int32),
                moves[:, 0],
                moves[:, 1]
            ], axis=-1)

            sim_boards = tf.tensor_scatter_nd_update(sim_boards, indices, players)
            immediate_winners = check_winner_tf(sim_boards)
            ongoing_mask = immediate_winners == 0

            sim_boards_to_simulate = tf.boolean_mask(sim_boards, ongoing_mask)
            players_to_simulate = 3 - tf.boolean_mask(players, ongoing_mask)
            winners_from_simulation = batch_random_simulation(sim_boards_to_simulate, players_to_simulate, self.board_size)

            final_winners = tf.where(ongoing_mask, tf.zeros_like(immediate_winners), immediate_winners)
            ongoing_indices = tf.where(ongoing_mask)
            final_winners = tf.tensor_scatter_nd_update(final_winners, ongoing_indices, winners_from_simulation)

            final_winners = tf.cast(final_winners, tf.int32)

            rewards = tf.where(final_winners == initial_players, 1.0,
                                tf.where(final_winners == 0, 0.0, -1.0))

            update_tensor = tf.tensor_scatter_nd_add(tf.zeros_like(move_scores), indices, rewards)
            move_scores += update_tensor

        best_moves_flat = tf.argmax(tf.reshape(move_scores, [batch_size, -1]), axis=1, output_type=tf.int32)
        best_moves = tf.stack([best_moves_flat // self.board_size, best_moves_flat % self.board_size], axis=-1)

        return best_moves

    @tf.function
    def sample_moves(self, boards):
        batch_size = tf.shape(boards)[0]
        empty_positions = tf.where(tf.equal(boards, 0))
        empty_positions = tf.cast(empty_positions, tf.int32)

        if tf.shape(empty_positions)[0] == 0:
            return tf.zeros([batch_size, 2], dtype=tf.int32)

        random_indices = tf.random.uniform([batch_size], minval=0, maxval=tf.shape(empty_positions)[0], dtype=tf.int32)
        selected_positions = tf.gather(empty_positions[:, 1:], random_indices)

        return selected_positions
