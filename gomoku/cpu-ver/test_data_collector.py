import unittest
import os
import json
import shutil
import numpy as np
from datetime import datetime
from gomoku_game import GomokuGame
from data_collector import DataCollector
from agents import RandomAgent

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

        # 模拟历史记录（用于训练数据提取测试）
        self.game.board_history = []
        for i in range(5):
            board = np.zeros((5, 5))
            # 每个历史记录添加一些棋子
            for j in range(i):
                board[j % 5, j // 5] = 1 if j % 2 == 0 else 2
            self.game.board_history.append(board)

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
            os.path.join(self.test_dir, "models"),
            os.path.join(self.test_dir, "training_data")  # 添加训练数据目录检查
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
    
    def test_save_game_with_parameters(self):
        """测试使用参数保存游戏"""
        parameters = {
            "board_size": 5,
            "win_condition": 5,
            "time_limit": 30
        }

        game_id, file_path = self.collector.save_game(
            self.game,
            metadata={"test": "test"},
            parameters=parameters
        )

        # 验证参数已保存
        with open(file_path, 'r') as f:
            game_data = json.load(f)

        self.assertIn("parameters", game_data)
        self.assertEqual(game_data["parameters"], parameters)

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

    def test_extract_training_data(self):
        """测试从游戏中提取训练数据"""
        # 创建有明确赢家的测试游戏
        game = GomokuGame(board_size=5)

        # 模拟黑棋获胜的棋局
        game.make_move(0, 0)  # 黑
        game.make_move(1, 0)  # 白
        game.make_move(0, 1)  # 黑
        game.make_move(1, 1)  # 白
        game.make_move(0, 2)  # 黑
        game.make_move(1, 2)  # 白
        game.make_move(0, 3)  # 黑
        game.make_move(1, 3)  # 白
        game.make_move(0, 4)  # 黑连成5子

        # 设置赢家
        game.winner = 1  # 黑棋获胜

        # 创建历史记录
        game.board_history = []
        current_board = np.zeros((5, 5))
        for i, move in enumerate(game.move_history):
            row, col, player = move
            new_board = current_board.copy()
            new_board[row, col] = player
            current_board = new_board
            game.board_history.append(new_board)

        # 提取训练数据
        training_data = self.collector.extract_training_data(game)

        # 验证数据
        self.assertGreater(len(training_data), 0)
        for sample in training_data:
            self.assertIn("board_state", sample)
            self.assertIn("move", sample)
            self.assertIn("player", sample)
            self.assertIn("label", sample)

        # 验证标签分配
        for sample in training_data:
            if sample["player"] == 1:  # 黑棋移动
                self.assertEqual(sample["label"], 1)  # 黑棋获胜，标签为1
            else:  # 白棋移动
                self.assertEqual(sample["label"], -1)  # 白棋失败，标签为-1

    def test_extract_training_data_draw(self):
        """测试从平局游戏中提取训练数据"""
        game = GomokuGame(board_size=3)
        # 设置平局
        game.winner = 0
        game.move_history = [(0, 0, 1), (0, 1, 2), (0, 2, 1)]
        game.board_history = [np.zeros((3, 3))]

        # 从平局游戏中提取训练数据
        training_data = self.collector.extract_training_data(game)

        # 验证平局游戏返回空列表
        self.assertEqual(len(training_data), 0)

    def test_save_training_data_to_npz(self):
        """测试保存训练数据为NPZ文件"""
        # 创建测试训练数据
        board_size = 5
        training_data = []
        for i in range(3):
            board = np.zeros((board_size, board_size))
            board[i, i] = 1  # 添加一些棋子
            sample = {
                "board_state": board,
                "move": (i, i+1),
                "player": 1 if i % 2 == 0 else 2,
                "label": 1 if i % 2 == 0 else -1
            }
            training_data.append(sample)

        # 保存为NPZ文件
        file_path = self.collector.save_training_data_to_npz(training_data)

        # 验证文件存在
        self.assertTrue(os.path.exists(file_path))

        # 加载NPZ文件并验证内容
        loaded_data = np.load(file_path)
        self.assertIn("boards", loaded_data)
        self.assertIn("moves", loaded_data)
        self.assertIn("labels", loaded_data)

        # 验证数据长度
        self.assertEqual(len(loaded_data["boards"]), 3)
        self.assertEqual(len(loaded_data["moves"]), 3)
        self.assertEqual(len(loaded_data["labels"]), 3)

        # 验证移动编码
        for i in range(3):
            expected_move = i * board_size + (i+1)
            self.assertEqual(loaded_data["moves"][i], expected_move)

    def test_empty_training_data_handling(self):
        """测试处理空训练数据的情况"""
        empty_data = []
        result = self.collector.save_training_data_to_npz(empty_data)
        self.assertIsNone(result)

    def test_custom_filename_for_training_data(self):
        """测试使用自定义文件名保存训练数据"""
        # 创建简单的训练数据
        training_data = [{
            "board_state": np.zeros((5, 5)),
            "move": (2, 2),
            "player": 1,
            "label": 1
        }]

        # 使用自定义文件名
        custom_name = "test_custom_name"
        file_path = self.collector.save_training_data_to_npz(training_data, file_name=custom_name)

        # 验证文件名
        expected_name = custom_name + ".npz" if not custom_name.endswith('.npz') else custom_name
        expected_path = os.path.join(self.test_dir, "training_data", expected_name)
        self.assertEqual(file_path, expected_path)
        self.assertTrue(os.path.exists(file_path))

    def test_generate_self_play_data(self):
        """测试自对弈数据生成"""
        # 模拟智能体
        class MockAgent:
            def __init__(self):
                self.__class__.__name__ = "MockAgent"

            def select_move(self, game):
                # 返回一个简单的有效移动
                for i in range(game.board_size):
                    for j in range(game.board_size):
                        if game.board[i][j] == 0:
                            return (i, j)
                return None

        # 使用模拟智能体
        mock_agents = [MockAgent(), MockAgent()]

        # 生成少量游戏用于测试
        training_data, stats_file = self.collector.generate_self_play_data(
            num_games=2,
            agents=mock_agents,
            visualize_interval=1
        )

        # 验证返回值
        self.assertIsNotNone(stats_file)
        self.assertTrue(os.path.exists(stats_file))

        # 验证统计文件
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        self.assertEqual(stats["num_games"], 2)
        self.assertEqual(len(stats["agents"]), 2)
        self.assertIn("results", stats)

        # 验证游戏文件已生成
        games_dir = os.path.join(self.test_dir, "games")
        game_files = [f for f in os.listdir(games_dir) if f.endswith(".json")]
        self.assertEqual(len(game_files), 2)

if __name__ == '__main__':
    unittest.main()
