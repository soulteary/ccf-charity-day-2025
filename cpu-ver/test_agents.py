import unittest
import numpy as np
from agents import RandomAgent, GreedyAgent, MCTSAgent
from gomoku_game import GomokuGame

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.game = GomokuGame(board_size=5)
    
    def test_random_agent(self):
        agent = RandomAgent()
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves(), "随机Agent选出的移动不合法")

    def test_greedy_agent_winning_move(self):
        agent = GreedyAgent()
        self.game.board = np.array([
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int8)
        self.game.current_player = 1
        move = agent.select_move(self.game)
        self.assertEqual(move, (0, 4), "贪心Agent未能选出必胜位置")

    def test_greedy_agent_block_move(self):
        agent = GreedyAgent()
        self.game.board = np.array([
            [2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int8)
        self.game.current_player = 1
        move = agent.select_move(self.game)
        self.assertEqual(move, (0, 4), "贪心Agent未能正确阻止对手胜利")

    def test_mcts_agent_valid_move(self):
        agent = MCTSAgent(iterations=50)
        move = agent.select_move(self.game)
        self.assertIn(move, self.game.get_valid_moves(), "MCTSAgent选出的移动不合法")

    def test_mcts_agent_winning_move(self):
        agent = MCTSAgent(iterations=100)
        self.game.board = np.array([
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int8)
        self.game.current_player = 1

        move = agent.select_move(self.game)
        success, msg = self.game.make_move(*move)
        self.assertTrue(success, f"落子失败: {msg}")
        self.assertTrue(self.game.check_win(*move), f"MCTSAgent 没有完成胜利落子，返回: {move}")

    def test_mcts_agent_block_move(self):
        agent = MCTSAgent(iterations=100)
        self.game.board = np.array([
            [2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int8)
        self.game.current_player = 1

        move = agent.select_move(self.game)
        success, msg = self.game.make_move(*move)
        self.assertTrue(success, f"落子失败: {msg}")

        # 模拟对手下一步，看是否还能立即胜
        self.game.current_player = 2
        winning_move_found = False
        for next_move in self.game.get_valid_moves():
            test_game = GomokuGame(self.game.board_size)
            test_game.board = self.game.board.copy()
            test_game.current_player = 2
            test_game.make_move(*next_move)
            if test_game.check_win(*next_move):
                winning_move_found = True
                break

        self.assertFalse(winning_move_found, f"MCTSAgent 未能有效防守，落子 {move} 后对手仍可立即获胜")


if __name__ == "__main__":
    unittest.main()
