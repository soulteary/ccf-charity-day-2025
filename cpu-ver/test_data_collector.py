import unittest
import os
import json
import shutil
import numpy as np
from datetime import datetime
from gomoku_game import GomokuGame
from data_collector import DataCollector

class TestDataCollector(unittest.TestCase):
    """单元测试DataCollector类"""
    
    def setUp(self):
        """测试前设置，创建临时测试目录"""
        self.test_dir = "test_gomoku_data"
        self.collector = DataCollector(base_dir=self.test_dir)
        
        # 创建测试用的游戏实例
        self.game = GomokuGame(board_size=5)
        
        # 模拟一些游戏移动
        self.game.make_move(1, 1)
        self.game.make_move(2, 2)
        self.game.make_move(1, 2)
        self.game.make_move(2, 3)
        
        # 添加模拟的visualize_board方法
        def mock_visualize_board(save_path=None):
            """模拟可视化方法，创建空文件"""
            if save_path:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write('mock visualization')
            return True
        
        # 添加模拟方法到游戏实例
        self.game.visualize_board = mock_visualize_board
    
    def tearDown(self):
        """测试后清理，移除测试目录"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_ensure_directories(self):
        """测试目录创建功能"""
        # 验证所有必要的目录都已创建
        expected_dirs = [
            self.test_dir,
            os.path.join(self.test_dir, "games"),
            os.path.join(self.test_dir, "visualizations"),
            os.path.join(self.test_dir, "statistics"),
            os.path.join(self.test_dir, "models")
        ]
        
        for directory in expected_dirs:
            self.assertTrue(os.path.exists(directory), f"目录 {directory} 未被创建")
    
    def test_save_game(self):
        """测试游戏保存功能"""
        metadata = {"player1": "Alice", "player2": "Bob", "time_limit": 30}
        game_id, file_path = self.collector.save_game(self.game, metadata)
        
        # 验证返回值
        self.assertIsNotNone(game_id)
        self.assertIsNotNone(file_path)
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(file_path), "游戏数据文件未创建")
        
        # 验证JSON内容
        with open(file_path, 'r') as f:
            game_data = json.load(f)
        
        self.assertEqual(game_data["game_id"], game_id)
        self.assertEqual(game_data["board_size"], 5)
        self.assertEqual(game_data["metadata"], metadata)
        self.assertEqual(len(game_data["move_history"]), 4)  # 我们执行了4次移动
        
        # 检查可视化文件
        vis_path = os.path.join(self.test_dir, "visualizations", f"{game_id}_final.png")
        self.assertTrue(os.path.exists(vis_path), "可视化文件未创建")
    
    def test_save_game_error_handling(self):
        """测试游戏保存的错误处理"""
        # 创建一个无法序列化的对象
        bad_metadata = {"function": lambda x: x}  # 函数无法被JSON序列化
        
        # 验证错误处理
        game_id, file_path = self.collector.save_game(self.game, bad_metadata)
        self.assertIsNone(game_id)
        self.assertIsNone(file_path)
    
    def test_load_game(self):
        """测试游戏加载功能"""
        # 先保存游戏
        game_id, file_path = self.collector.save_game(self.game)
        
        # 通过ID加载
        loaded_data = self.collector.load_game(game_id=game_id)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["game_id"], game_id)
        self.assertEqual(loaded_data["board_size"], 5)
        
        # 通过文件路径加载
        loaded_data = self.collector.load_game(file_path=file_path)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data["game_id"], game_id)
    
    def test_load_game_error_handling(self):
        """测试游戏加载的错误处理"""
        # 不存在的游戏ID
        loaded_data = self.collector.load_game(game_id="nonexistent_game")
        self.assertIsNone(loaded_data)
        
        # 无效的文件路径
        loaded_data = self.collector.load_game(file_path="invalid/path.json")
        self.assertIsNone(loaded_data)
        
        # 没有提供ID或路径
        with self.assertRaises(ValueError):
            self.collector.load_game()
    
    def test_collect_statistics(self):
        """测试统计信息收集功能"""
        # 创建几个测试游戏并保存
        for i in range(3):
            game = GomokuGame(board_size=5)
            
            # 对每个游戏添加不同数量的移动以匹配7.0的平均值
            # 第一个游戏：7次移动
            if i == 0:
                for j in range(7):
                    row, col = j % 5, (j // 5) % 5
                    game.make_move(row, col)
            # 第二个游戏：6次移动
            elif i == 1:
                for j in range(6):
                    row, col = j % 5, (j // 5) % 5
                    game.make_move(row, col)
            # 第三个游戏：8次移动
            else:
                for j in range(8):
                    row, col = j % 5, (j // 5) % 5
                    game.make_move(row, col)
            
            # 设置不同的赢家
            game.winner = i  # 0=平局, 1=玩家1赢, 2=玩家2赢
            
            # 添加visualize_board方法
            game.visualize_board = self.game.visualize_board
            
            # 保存游戏
            self.collector.save_game(game)
        
        # 收集统计信息
        stats = self.collector.collect_statistics()
        
        # 打印实际值进行调试
        print(f"实际平均移动数: {stats['avg_moves']}")
        
        # 验证统计信息
        self.assertEqual(stats["total_games"], 3)
        self.assertEqual(stats["player1_wins"], 1)
        self.assertEqual(stats["player2_wins"], 1)
        self.assertEqual(stats["draws"], 1)
        self.assertEqual(stats["avg_moves"], 7.0)  # 总计21次移动 / 3个游戏 = 7.0
        
        # 验证统计文件已保存
        stats_dir = os.path.join(self.test_dir, "statistics")
        stats_files = [f for f in os.listdir(stats_dir) if f.endswith(".json")]
        self.assertTrue(len(stats_files) > 0, "统计文件未创建")

if __name__ == '__main__':
    unittest.main()