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

        # Ensure initial_boards and initial_players are of type int32
        initial_boards = tf.cast(initial_boards, tf.int32)  # Convert to int32
        initial_players = tf.cast(initial_players, tf.int32)  # Convert to int32

        for _ in tf.range(self.iterations):
            moves = self.sample_moves(initial_boards)
            sim_boards = tf.identity(initial_boards)
            players = tf.identity(initial_players)

            # Ensure sim_boards and players are of the same type
            sim_boards = tf.cast(sim_boards, tf.int32)  # Ensure sim_boards is int32

            indices = tf.stack([
                tf.range(batch_size, dtype=tf.int32),
                moves[:, 0],
                moves[:, 1]
            ], axis=-1)

            # Now both sim_boards and players are int32, we can safely perform the scatter update
            sim_boards = tf.tensor_scatter_nd_update(sim_boards, indices, players)
            immediate_winners = check_winner_tf(sim_boards)
            ongoing_mask = immediate_winners == 0

            sim_boards_to_simulate = tf.boolean_mask(sim_boards, ongoing_mask)
            players_to_simulate = 3 - tf.boolean_mask(players, ongoing_mask)
            winners_from_simulation = batch_random_simulation(sim_boards_to_simulate, players_to_simulate, self.board_size)

            final_winners = tf.where(ongoing_mask, tf.zeros_like(immediate_winners), immediate_winners)
            ongoing_indices = tf.where(ongoing_mask)
            final_winners = tf.tensor_scatter_nd_update(final_winners, ongoing_indices, winners_from_simulation)

            # 强制类型统一为 int32，确保类型一致
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
        empty_positions = tf.where(tf.equal(boards, 0))  # 获取所有空位的位置
        empty_positions = tf.cast(empty_positions, tf.int32)  # 确保 empty_positions 为 int32 类型
        moves = tf.TensorArray(tf.int32, size=batch_size)

        for idx in tf.range(batch_size):
            # 将 idx 转换为 int32 类型以与 empty_positions[:, 0] 进行比较
            idx_32 = tf.cast(idx, tf.int32)

            # 确保 empty_positions[:, 0] 是 int32 类型，避免类型不一致
            board_empty = empty_positions[empty_positions[:, 0] == idx_32][:, 1:]
            
            # 检查 board_empty 的形状
            num_empty = tf.shape(board_empty)[0]

            # 如果没有空位，返回一个默认位置
            if num_empty == 0:
                chosen_move = tf.constant([0, 0], dtype=tf.int32)  # 默认返回 (0, 0) 位置
            else:
                # 随机选择一个位置
                random_idx = tf.random.uniform([], maxval=num_empty, dtype=tf.int32)
                chosen_move = tf.cast(board_empty[random_idx], tf.int32)

            # 将选择的移动位置存储
            moves = moves.write(idx, chosen_move)

        return moves.stack()
