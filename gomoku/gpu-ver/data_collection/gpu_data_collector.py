import os
import json
from datetime import datetime

class GPUDataCollector:
    def __init__(self, base_dir="gpu_gomoku_data"):
        self.base_dir = base_dir
        self.setup_directories()

    def setup_directories(self):
        try:
            for sub_dir in ["games", "visualizations", "statistics", "configs"]:
                os.makedirs(os.path.join(self.base_dir, sub_dir), exist_ok=True)
        except OSError as e:
            print(f"Error creating directories: {e}")
            raise

    def save_game_data(self, game_id, game_history, winner, config):
        assert isinstance(game_history, list), "game_history must be a list"
        assert isinstance(config, dict), "config must be a dict"
        try:
            game_data = {
                "game_id": game_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "winner": winner,
                "move_history": game_history
            }
            game_file = os.path.join(self.base_dir, "games", f"{game_id}.json")
            config_file = os.path.join(self.base_dir, "configs", f"{game_id}_config.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f, indent=4)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving game data: {e}")
            raise

    def save_statistics(self, statistics):
        assert isinstance(statistics, dict), "statistics must be a dict"
        stats_file = os.path.join(
            self.base_dir, "statistics",
            f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(stats_file, 'w') as f:
                json.dump(statistics, f, indent=4)
            return stats_file
        except IOError as e:
            print(f"Error saving statistics: {e}")
            raise
