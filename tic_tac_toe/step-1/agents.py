import random
from tic_tac_toe import TicTacToeGame

class RandomAgent:
    def select_move(self, game):
        moves = game.get_valid_moves()
        return random.choice(moves) if moves else None

class GreedyAgent:
    def select_move(self, game):
        for move in game.get_valid_moves():
            test_game = TicTacToeGame()
            test_game.board = game.board.copy()
            test_game.current_player = game.current_player
            test_game.make_move(*move)
            if test_game.winner == game.current_player:
                return move

        return RandomAgent().select_move(game)
