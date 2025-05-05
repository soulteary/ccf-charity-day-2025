import unittest
from tic_tac_toe import TicTacToeGame

class TestTicTacToeGame(unittest.TestCase):
    def test_initialization(self):
        game = TicTacToeGame()
        self.assertEqual(game.board_size, 3)
        self.assertEqual(game.current_player, 1)
        self.assertFalse(game.game_over)
        self.assertIsNone(game.winner)

    def test_make_valid_move(self):
        game = TicTacToeGame()
        valid = game.make_move(0, 0)
        self.assertTrue(valid)
        self.assertEqual(game.board[0, 0], 1)
        self.assertFalse(game.game_over)

    def test_make_invalid_move(self):
        game = TicTacToeGame()
        game.make_move(0, 0)
        valid = game.make_move(0, 0)
        self.assertFalse(valid)

    def test_winner_detection_row(self):
        game = TicTacToeGame()
        moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
        for move in moves:
            game.make_move(*move)
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

    def test_winner_detection_column(self):
        game = TicTacToeGame()
        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        for move in moves:
            game.make_move(*move)
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

    def test_winner_detection_diagonal(self):
        game = TicTacToeGame()
        moves = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 2)]
        for move in moves:
            game.make_move(*move)
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

    def test_draw(self):
        game = TicTacToeGame()
        moves = [(0, 0), (0, 1), (0, 2),
                 (1, 1), (1, 0), (1, 2),
                 (2, 1), (2, 0), (2, 2)]
        players = [1, 2, 1, 2, 1, 2, 1, 2, 1]
        for player, move in zip(players, moves):
            game.current_player = player
            game.make_move(*move)
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 0)

    def test_copy_game(self):
        game = TicTacToeGame()
        game.make_move(0, 0)
        copied_game = game.copy()
        self.assertEqual(copied_game.board[0, 0], 1)
        self.assertEqual(copied_game.current_player, 2)
        self.assertFalse(copied_game.game_over)
        copied_game.make_move(1, 1)
        self.assertEqual(copied_game.board[1, 1], 2)
        self.assertEqual(game.board[1, 1], 0)  # 原游戏未改变，保证独立复制

if __name__ == "__main__":
    unittest.main()
