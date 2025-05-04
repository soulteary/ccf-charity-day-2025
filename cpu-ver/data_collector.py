import os
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from gomoku_game import GomokuGame
from agents import RandomAgent, GreedyAgent, MCTSAgent

class DataCollector:
    """五子棋数据收集器"""

    def __init__(self, base_dir="gomoku_data"):
        self.base_dir = base_dir
        self.ensure_directories()

    def ensure_directories(self):
        """确保必要的目录结构存在"""
        dirs = [
            self.base_dir,
            os.path.join(self.base_dir, "games"),
            os.path.join(self.base_dir, "visualizations"),
            os.path.join(self.base_dir, "statistics"),
            os.path.join(self.base_dir, "models"),
            os.path.join(self.base_dir, "training_data")
        ]

        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def save_game(self, game, metadata=None, parameters=None):
        """保存游戏数据，包括棋局信息和元数据

        Args:
            game: GomokuGame实例
            metadata: 额外的元数据字典
            parameters: 游戏生成参数，通常来自命令行参数

        Returns:
            tuple: (game_id, file_path) 游戏ID和保存路径
        """
        if metadata is None:
            metadata = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = f"game_{timestamp}_{random.randint(1000, 9999)}"

        game_data = {
            "game_id": game_id,
            "timestamp": timestamp,
            "metadata": metadata,
            "board_size": game.board_size,
            "winner": game.winner,
            "move_history": game.move_history,
            "final_board": game.board.tolist()
        }

        # 添加游戏生成参数信息，如果有的话
        if parameters is not None:
            game_data["parameters"] = parameters

        try:
            file_path = os.path.join(self.base_dir, "games", f"{game_id}.json")
            with open(file_path, 'w') as f:
                json.dump(game_data, f, indent=4)
            
            # 如果游戏对象有visualize_board方法，则使用它
            vis_path = os.path.join(self.base_dir, "visualizations", f"{game_id}_final.png")
            if hasattr(game, 'visualize_board') and callable(game.visualize_board):
                game.visualize_board(save_path=vis_path)

            return game_id, file_path
        except Exception as e:
            print(f"Error saving game: {e}")
            return None, None

    def load_game(self, game_id=None, file_path=None):
        """加载保存的游戏数据

        Args:
            game_id: 游戏ID
            file_path: 或者直接提供文件路径

        Returns:
            dict: 游戏数据
        """
        if file_path is None and game_id is not None:
            file_path = os.path.join(self.base_dir, "games", f"{game_id}.json")

        if file_path is None:
            raise ValueError("Either game_id or file_path must be provided")

        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            return game_data
        except Exception as e:
            print(f"Error loading game: {e}")
            return None

    def collect_statistics(self, games_data=None):
        """收集游戏统计信息

        Args:
            games_data: 游戏数据列表，如果为None则加载所有

        Returns:
            dict: 统计信息
        """
        if games_data is None:
            games_data = []
            games_dir = os.path.join(self.base_dir, "games")
            if os.path.exists(games_dir):
                for filename in os.listdir(games_dir):
                    if filename.endswith(".json"):
                        file_path = os.path.join(games_dir, filename)
                        game_data = self.load_game(file_path=file_path)
                        if game_data:
                            games_data.append(game_data)

        # 基本统计
        stats = {
            "total_games": len(games_data),
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "avg_moves": 0,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        total_moves = 0
        for game in games_data:
            if game["winner"] == 1:
                stats["player1_wins"] += 1
            elif game["winner"] == 2:
                stats["player2_wins"] += 1
            elif game["winner"] == 0:
                stats["draws"] += 1

            total_moves += len(game["move_history"])

        if stats["total_games"] > 0:
            stats["avg_moves"] = total_moves / stats["total_games"]

        # 保存统计信息
        stats_path = os.path.join(self.base_dir, "statistics", 
                                 f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Error saving statistics: {e}")

        return stats

    def extract_training_data(self, game):
        """从游戏中提取训练数据

        Args:
            game: GomokuGame实例

        Returns:
            list: 包含训练样本的列表
        """
        training_data = []
        winner = game.winner
        move_history = game.move_history
        board_history = game.board_history

        # 如果没有赢家（平局）或者没有历史记录，则返回空列表
        if winner == 0 or not move_history or not board_history:
            return []

        # 对每一步棋，创建一个训练样本
        for move_idx, move in enumerate(move_history):
            row, col, player = move

            # 棋盘状态是这一步棋之前的状态
            if move_idx < len(board_history) - 1:  # 确保有足够的历史记录
                board_state = board_history[move_idx].copy()

                # 根据游戏结果分配标签
                # 如果这个玩家是赢家，标签为1；否则标签为-1
                label = 1 if player == winner else -1

                sample = {
                    "board_state": board_state,
                    "move": (row, col),
                    "player": player,
                    "label": label
                }

                training_data.append(sample)

        return training_data

    def generate_self_play_data(self, num_games=100, agents=None, visualize_interval=10):
        """生成自对弈数据（仅生成比赛数据，不生成NPZ训练文件）

        Args:
            num_games: 要生成的游戏数量
            agents: 要使用的智能体列表，如果为None则使用默认智能体
            visualize_interval: 打印进度的间隔

        Returns:
            tuple: (训练数据列表, 统计文件路径)
        """
        if agents is None:
            agents = [GreedyAgent(), MCTSAgent(iterations=200)]

        all_training_data = []
        game_results = {"black_wins": 0, "white_wins": 0, "draws": 0}

        for game_idx in tqdm(range(num_games), desc="生成自对弈数据"):
            game = GomokuGame()
            agent_idx = 0

            while not game.game_over:
                agent = agents[agent_idx % len(agents)]
                move = agent.select_move(game)

                if move:
                    success, message = game.make_move(move[0], move[1])
                    if not success:
                        print(f"移动失败: {message}")
                        break
                else:
                    print("没有有效的移动")
                    break

                agent_idx += 1

            # 记录游戏结果
            if game.winner == 1:
                game_results["black_wins"] += 1
            elif game.winner == 2:
                game_results["white_wins"] += 1
            else:
                game_results["draws"] += 1

            # 保存游戏数据
            metadata = {
                "agents": [agent.__class__.__name__ for agent in agents],
                "result": "黑胜" if game.winner == 1 else ("白胜" if game.winner == 2 else "平局")
            }

            game_id, _ = self.save_game(game, metadata)

            # 提取训练数据
            training_data = self.extract_training_data(game)
            all_training_data.extend(training_data)

            # 定期打印进度
            if (game_idx + 1) % visualize_interval == 0:
                print(f"已完成 {game_idx + 1}/{num_games} 场对弈，"
                      f"黑胜: {game_results['black_wins']}, "
                      f"白胜: {game_results['white_wins']}, "
                      f"平局: {game_results['draws']}")

        # 保存汇总统计信息
        stats_file = os.path.join(self.base_dir, "statistics", f"self_play_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(stats_file, 'w') as f:
            json.dump({
                "num_games": num_games,
                "agents": [agent.__class__.__name__ for agent in agents],
                "results": game_results,
                "num_training_samples": len(all_training_data)
            }, f, indent=4)

        return all_training_data, stats_file

    def save_training_data_to_npz(self, training_data, file_name=None):
        """将训练数据保存为NPZ文件，可在采集完数据后手动调用

        Args:
            training_data: 从generate_self_play_data生成的训练数据列表
            file_name: 自定义文件名（可选）

        Returns:
            str: NPZ文件路径
        """
        if not training_data:
            print("没有训练数据可保存")
            return None

        # 创建保存目录
        training_data_dir = os.path.join(self.base_dir, "training_data")
        os.makedirs(training_data_dir, exist_ok=True)

        # 生成文件名
        if file_name is None:
            file_name = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        elif not file_name.endswith('.npz'):
            file_name += '.npz'

        training_data_file = os.path.join(training_data_dir, file_name)

        # 转换为NumPy数组
        boards = []
        moves = []
        labels = []

        for sample in training_data:
            boards.append(sample["board_state"])
            move = sample["move"]
            # 使用一维编码表示移动位置
            board_size = sample["board_state"].shape[0]
            move_encoded = move[0] * board_size + move[1]
            moves.append(move_encoded)
            labels.append(sample["label"])

        # 保存为NPZ文件
        np.savez(
            training_data_file,
            boards=np.array(boards),
            moves=np.array(moves),
            labels=np.array(labels)
        )

        print(f"保存了 {len(training_data)} 个训练样本到 {training_data_file}")

        return training_data_file
