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
        if self.check_winner(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 3 - self.current_player

        return True

    def check_winner(self, row, col):
        player = self.board[row, col]
        row_win = np.all(self.board[row, :] == player)
        col_win = np.all(self.board[:, col] == player)
        diag_win = row == col and np.all(np.diag(self.board) == player)
        anti_diag_win = row + col == self.board_size - 1 and np.all(np.diag(np.fliplr(self.board)) == player)
        return row_win or col_win or diag_win or anti_diag_win

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == 0)))

    def copy(self):
        new_game = TicTacToeGame()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game

    def __str__(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(symbols[cell] for cell in row) for row in self.board]
        return "\n".join(rows)
