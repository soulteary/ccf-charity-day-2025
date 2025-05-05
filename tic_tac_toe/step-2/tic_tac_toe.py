import numpy as np

class TicTacToeGame:
    def __init__(self):
        self.board_size = 3
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def make_move(self, row, col):
        if self.game_over or self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player
        self.check_winner(row, col)
        self.current_player = 3 - self.current_player  # Switch players
        return True

    def check_winner(self, row, col):
        if (np.all(self.board[row, :] == self.current_player) or
            np.all(self.board[:, col] == self.current_player) or
            (row == col and np.all(np.diag(self.board) == self.current_player)) or
            (row + col == self.board_size - 1 and np.all(np.diag(np.fliplr(self.board)) == self.current_player))):
            self.game_over = True
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.game_over = True
            self.winner = 0  # Draw

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def __str__(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(symbols[cell] for cell in row) for row in self.board]
        return "\n".join(rows)
