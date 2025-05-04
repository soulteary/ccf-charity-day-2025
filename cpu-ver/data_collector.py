import os
import json
import random
from datetime import datetime
from gomoku_game import GomokuGame

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
            os.path.join(self.base_dir, "models")
        ]
        
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def save_game(self, game, metadata=None):
        """保存游戏数据，包括棋局信息和元数据
        
        Args:
            game: GomokuGame实例
            metadata: 额外的元数据字典
            
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