import numpy as np

def check_winner(board):
    size = board.shape[0]
    for player in [1, 2]:
        for i in range(size):
            for j in range(size - 4):
                if np.all(board[i, j:j + 5] == player):
                    return player
        for i in range(size - 4):
            for j in range(size):
                if np.all(board[i:i + 5, j] == player):
                    return player
        for i in range(size - 4):
            for j in range(size - 4):
                if np.all([board[i + k, j + k] == player for k in range(5)]):
                    return player
        for i in range(4, size):
            for j in range(size - 4):
                if np.all([board[i - k, j + k] == player for k in range(5)]):
                    return player
    if np.all(board != 0):
        return -1
    return 0
