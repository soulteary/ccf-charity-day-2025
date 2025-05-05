import unittest
from tic_tac_toe import TicTacToeGame
from agents import RandomAgent, GreedyAgent, MCTSAgent

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
        self.assertEqual(move, (0, 2))

    def test_greedy_agent_random_move_when_no_win(self):
        agent = GreedyAgent()
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves())

    def test_mcts_agent_select_move(self):
        agent = MCTSAgent(simulations=100)
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves())

    def test_mcts_agent_winning_move(self):
        agent = MCTSAgent(simulations=5000)
        game = TicTacToeGame()
        moves = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for move in moves:
            game.make_move(*move)

        move = agent.select_move(game)

        self.assertEqual(move, (0, 2), f"MCTS未选择明确获胜位置，当前选择: {move}")

        test_game = game.copy()
        valid_move = test_game.make_move(*move)

        self.assertTrue(valid_move, f"MCTS选择非法位置: {move}")
        self.assertTrue(test_game.game_over, "游戏未结束，MCTS未找到结束局势的步骤")
        self.assertEqual(test_game.winner, game.current_player,
                        f"MCTS未选择导致获胜的位置, 当前选择: {move}")


if __name__ == "__main__":
    unittest.main()
