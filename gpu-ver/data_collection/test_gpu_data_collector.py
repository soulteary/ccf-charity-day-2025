import unittest
import os
import shutil
import json
from .gpu_data_collector import GPUDataCollector

class TestGPUDataCollector(unittest.TestCase):

    def setUp(self):
        self.test_dir = "test_gpu_data"
        self.collector = GPUDataCollector(base_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_setup_directories(self):
        expected_dirs = ["games", "visualizations", "statistics", "configs"]
        for sub_dir in expected_dirs:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, sub_dir)))

    def test_save_game_data(self):
        game_id = "test_game_001"
        game_history = [{"move": [7, 7], "player": 1}, {"move": [7, 8], "player": 2}]
        winner = 1
        config = {"board_size": 15, "players": ["AI1", "AI2"]}

        self.collector.save_game_data(game_id, game_history, winner, config)

        game_file = os.path.join(self.test_dir, "games", f"{game_id}.json")
        config_file = os.path.join(self.test_dir, "configs", f"{game_id}_config.json")

        self.assertTrue(os.path.isfile(game_file))
        self.assertTrue(os.path.isfile(config_file))

        with open(game_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["game_id"], game_id)
            self.assertEqual(data["winner"], winner)
            self.assertEqual(data["move_history"], game_history)

        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            self.assertEqual(loaded_config, config)

    def test_save_statistics(self):
        statistics = {"wins": { "AI1": 10, "AI2": 15 }, "total_games": 25}
        stats_file = self.collector.save_statistics(statistics)

        self.assertTrue(os.path.isfile(stats_file))

        with open(stats_file, 'r') as f:
            loaded_stats = json.load(f)
            self.assertEqual(loaded_stats, statistics)

    def test_invalid_game_history(self):
        with self.assertRaises(AssertionError):
            self.collector.save_game_data("game_id", "invalid_history", 1, {})

    def test_invalid_config(self):
        with self.assertRaises(AssertionError):
            self.collector.save_game_data("game_id", [], 1, "invalid_config")

    def test_invalid_statistics(self):
        with self.assertRaises(AssertionError):
            self.collector.save_statistics("invalid_statistics")

if __name__ == '__main__':
    unittest.main()
