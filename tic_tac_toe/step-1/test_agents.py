import unittest
from tic_tac_toe import TicTacToeGame
from agents import RandomAgent, GreedyAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToeGame()

    def test_random_agent_select_move(self):
        agent = RandomAgent()
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves())

    def test_greedy_agent_winning_move(self):
        agent = GreedyAgent()
        moves = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for move in moves:
            self.game.make_move(*move)
        move = agent.select_move(self.game)
        self.assertEqual(move, (0, 2))  # 胜利的下一步

    def test_greedy_agent_random_move_when_no_win(self):
        agent = GreedyAgent()
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves())

if __name__ == "__main__":
    unittest.main()
