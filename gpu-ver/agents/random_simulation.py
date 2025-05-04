import tensorflow as tf
from environment.utils import check_winner_tf

@tf.function
def batch_random_simulation(boards, players, board_size):
    batch_size = tf.shape(boards)[0]
    done = tf.zeros(batch_size, dtype=tf.bool)
    winner = tf.zeros(batch_size, dtype=tf.int32)

    for _ in tf.range(board_size * board_size):
        empty_mask = tf.equal(boards, 0)
        moves_left = tf.reduce_any(empty_mask, axis=[1, 2])

        if not tf.reduce_any(moves_left & ~done):
            break

        random_values = tf.random.uniform((batch_size, board_size, board_size), dtype=tf.float32)
        masked_values = tf.where(empty_mask, random_values, -1.0)

        flat_indices = tf.argmax(tf.reshape(masked_values, [batch_size, -1]), axis=1, output_type=tf.int32)
        rows, cols = flat_indices // board_size, flat_indices % board_size
        indices = tf.stack([tf.range(batch_size), rows, cols], axis=-1)

        updates = tf.where(~done, players, 0)
        boards = tf.tensor_scatter_nd_update(boards, indices, updates)

        current_winner = check_winner_tf(boards)
        new_winner_mask = (winner == 0) & (current_winner > 0)
        winner = tf.where(new_winner_mask, current_winner, winner)
        done |= (current_winner > 0)

        players = 3 - players

    return winner
