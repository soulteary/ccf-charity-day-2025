import unittest
import numpy as np
from gomoku_game import GomokuGame

class TestGomokuGame(unittest.TestCase):
    """五子棋游戏逻辑单元测试"""
    
    def setUp(self):
        """每个测试前初始化"""
        self.game = GomokuGame(board_size=15)
        self.small_game = GomokuGame(board_size=5)
    
    def test_initialization(self):
        """测试游戏初始化"""
        # 测试默认构造函数
        default_game = GomokuGame()
        self.assertEqual(default_game.board_size, 15)
        self.assertEqual(default_game.current_player, 1)
        self.assertFalse(default_game.game_over)
        self.assertIsNone(default_game.winner)
        self.assertEqual(len(default_game.move_history), 0)
        self.assertEqual(len(default_game.board_history), 1)  # 初始棋盘状态
        
        # 测试自定义棋盘大小
        custom_game = GomokuGame(board_size=9)
        self.assertEqual(custom_game.board_size, 9)
        self.assertEqual(custom_game.board.shape, (9, 9))
    
    def test_reset(self):
        """测试游戏重置功能"""
        # 先进行一些移动
        self.game.make_move(7, 7)
        self.game.make_move(8, 8)
        
        # 重置游戏
        state = self.game.reset()
        
        # 验证重置后的状态
        self.assertEqual(self.game.current_player, 1)
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)
        self.assertEqual(len(self.game.move_history), 0)
        self.assertEqual(len(self.game.board_history), 1)
        self.assertTrue(np.array_equal(self.game.board, np.zeros((15, 15), dtype=np.int8)))
        
        # 验证返回的状态
        self.assertTrue(np.array_equal(state['board'], np.zeros((15, 15), dtype=np.int8)))
        self.assertEqual(state['current_player'], 1)
        self.assertFalse(state['game_over'])
        self.assertIsNone(state['winner'])
        self.assertEqual(len(state['move_history']), 0)
    
    def test_get_valid_moves(self):
        """测试获取有效移动"""
        # 新游戏所有位置都有效
        valid_moves = self.small_game.get_valid_moves()
        self.assertEqual(len(valid_moves), 25)  # 5x5 棋盘有25个有效位置
        
        # 放置一些棋子后验证
        self.small_game.make_move(2, 2)
        self.small_game.make_move(2, 3)
        
        valid_moves = self.small_game.get_valid_moves()
        self.assertEqual(len(valid_moves), 23)  # 25 - 2 = 23个有效位置
        self.assertNotIn((2, 2), valid_moves)
        self.assertNotIn((2, 3), valid_moves)
        
        # 游戏结束后没有有效移动
        self.small_game.game_over = True
        valid_moves = self.small_game.get_valid_moves()
        self.assertEqual(len(valid_moves), 0)
    
    def test_make_move(self):
        """测试执行移动"""
        # 测试合法移动
        success, message = self.game.make_move(7, 7)
        self.assertTrue(success)
        self.assertEqual(self.game.board[7, 7], 1)
        self.assertEqual(self.game.current_player, 2)  # 切换玩家
        self.assertEqual(len(self.game.move_history), 1)
        self.assertEqual(len(self.game.board_history), 2)
        
        # 测试重复移动
        success, message = self.game.make_move(7, 7)
        self.assertFalse(success)
        self.assertEqual(message, "该位置已有棋子")
        
        # 测试超出范围
        success, message = self.game.make_move(15, 15)
        self.assertFalse(success)
        self.assertEqual(message, "移动超出棋盘范围")
        
        # 测试游戏结束后移动
        self.game.game_over = True
        success, message = self.game.make_move(8, 8)
        self.assertFalse(success)
        self.assertEqual(message, "游戏已结束")
    
    def test_check_win_horizontal(self):
        """测试水平方向胜利检测"""
        # 第一个玩家在第7行连续放置5个棋子
        for i in range(5):
            if i < 4:
                success, _ = self.game.make_move(7, i)
                self.assertTrue(success)
                self.assertFalse(self.game.game_over)
                
                # 第二个玩家在第8行放置棋子
                success, _ = self.game.make_move(8, i)
                self.assertTrue(success)
            else:
                # 第五个棋子应该导致游戏结束
                success, message = self.game.make_move(7, i)
                self.assertTrue(success)
                self.assertTrue(self.game.game_over)
                self.assertEqual(self.game.winner, 1)
                self.assertEqual(message, "玩家 1 获胜")
    
    def test_check_win_vertical(self):
        """测试垂直方向胜利检测"""
        # 第一个玩家在第7列连续放置5个棋子
        for i in range(5):
            if i < 4:
                success, _ = self.game.make_move(i, 7)
                self.assertTrue(success)
                self.assertFalse(self.game.game_over)
                
                # 第二个玩家在第8列放置棋子
                success, _ = self.game.make_move(i, 8)
                self.assertTrue(success)
            else:
                # 第五个棋子应该导致游戏结束
                success, message = self.game.make_move(i, 7)
                self.assertTrue(success)
                self.assertTrue(self.game.game_over)
                self.assertEqual(self.game.winner, 1)
                self.assertEqual(message, "玩家 1 获胜")
    
    def test_check_win_diagonal(self):
        """测试对角线方向胜利检测"""
        # 主对角线
        for i in range(5):
            if i < 4:
                success, _ = self.game.make_move(i, i)
                self.assertTrue(success)
                self.assertFalse(self.game.game_over)
                
                # 第二个玩家在副对角线放置棋子
                success, _ = self.game.make_move(i, 14-i)
                self.assertTrue(success)
            else:
                # 第五个棋子应该导致游戏结束
                success, message = self.game.make_move(i, i)
                self.assertTrue(success)
                self.assertTrue(self.game.game_over)
                self.assertEqual(self.game.winner, 1)
                self.assertEqual(message, "玩家 1 获胜")
    
    def test_check_win_antidiagonal(self):
        """测试反对角线方向胜利检测"""
        # 副对角线
        for i in range(5):
            if i < 4:
                success, _ = self.game.make_move(i, 4-i)
                self.assertTrue(success)
                self.assertFalse(self.game.game_over)
                
                # 第二个玩家在其他位置放置棋子
                success, _ = self.game.make_move(i, 5)
                self.assertTrue(success)
            else:
                # 第五个棋子应该导致游戏结束
                success, message = self.game.make_move(i, 4-i)
                self.assertTrue(success)
                self.assertTrue(self.game.game_over)
                self.assertEqual(self.game.winner, 1)
                self.assertEqual(message, "玩家 1 获胜")
    
    def test_draw(self):
        """测试平局情况"""
        # 使用小棋盘便于测试
        game = GomokuGame(board_size=3)
        
        # 填满棋盘但不构成胜局
        moves = [
            (0, 0), (0, 1),
            (0, 2), (1, 0),
            (1, 1), (1, 2),
            (2, 1), (2, 0),
        ]
        
        for i, (row, col) in enumerate(moves):
            success, _ = game.make_move(row, col)
            self.assertTrue(success)
            self.assertFalse(game.game_over)
        
        # 最后一步应该导致平局
        success, message = game.make_move(2, 2)
        self.assertTrue(success)
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 0)  # 0表示平局
        self.assertEqual(message, "平局")
    
    def test_get_state(self):
        """测试获取游戏状态"""
        # 进行一些移动
        self.game.make_move(7, 7)
        self.game.make_move(8, 8)
        
        # 获取状态
        state = self.game.get_state()
        
        # 验证状态
        self.assertTrue(np.array_equal(state['board'], self.game.board))
        self.assertEqual(state['current_player'], self.game.current_player)
        self.assertEqual(state['game_over'], self.game.game_over)
        self.assertEqual(state['winner'], self.game.winner)
        self.assertEqual(state['move_history'], self.game.move_history)
        
        # 验证状态是副本而非引用
        state['board'][0, 0] = 9
        self.assertNotEqual(state['board'][0, 0], self.game.board[0, 0])
    
    def test_save_board_state(self):
        """测试保存棋盘状态"""
        # 初始状态
        self.assertEqual(len(self.game.board_history), 1)
        
        # 进行移动后验证
        self.game.make_move(7, 7)
        self.assertEqual(len(self.game.board_history), 2)
        self.assertEqual(self.game.board_history[-1][7, 7], 1)
        
        self.game.make_move(8, 8)
        self.assertEqual(len(self.game.board_history), 3)
        self.assertEqual(self.game.board_history[-1][8, 8], 2)
        
        # 验证历史记录的独立性
        self.game.board_history[1][7, 7] = 0
        self.assertNotEqual(self.game.board_history[1][7, 7], self.game.board_history[2][7, 7])

if __name__ == '__main__':
    unittest.main()