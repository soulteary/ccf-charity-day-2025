import tensorflow as tf

@tf.function
def check_winner_tf(boards):
    batch_size = tf.shape(boards)[0]
    winner = tf.zeros((batch_size,), dtype=tf.int32)

    def conv_check(board, player, kernel):
        board_exp = tf.cast(tf.equal(board, player), tf.float32)[tf.newaxis, ..., tf.newaxis]
        conv = tf.nn.conv2d(board_exp, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return tf.reduce_any(tf.equal(conv, 5))

    kernels = {
        'horizontal': tf.ones((1, 5, 1, 1), dtype=tf.float32),
        'vertical': tf.ones((5, 1, 1, 1), dtype=tf.float32),
        'diag_main': tf.reshape(tf.eye(5, dtype=tf.float32), (5, 5, 1, 1)),
        'diag_anti': tf.reshape(tf.reverse(tf.eye(5, dtype=tf.float32), axis=[1]), (5, 5, 1, 1))
    }

    for player in [1, 2]:
        mask_horizontal = tf.vectorized_map(lambda board: conv_check(board, player, kernels['horizontal']), boards)
        mask_vertical = tf.vectorized_map(lambda board: conv_check(board, player, kernels['vertical']), boards)
        mask_diag_main = tf.vectorized_map(lambda board: conv_check(board, player, kernels['diag_main']), boards)
        mask_diag_anti = tf.vectorized_map(lambda board: conv_check(board, player, kernels['diag_anti']), boards)

        mask = mask_horizontal | mask_vertical | mask_diag_main | mask_diag_anti
        winner = tf.where(mask, player, winner)

    return winner
