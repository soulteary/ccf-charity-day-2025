import unittest
import os
import json
import shutil
from tic_tac_toe import TicTacToeGame
from data_collector import DataCollector

class TestDataCollector(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_tic_tac_toe_data"
        self.collector = DataCollector(base_dir=self.test_dir)
        self.game = TicTacToeGame()
        self.game.make_move(0, 0)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_game(self):
        metadata = {"test_case": "unit_test"}
        self.collector.save_game(self.game, metadata)

        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 1)

        file_path = os.path.join(self.test_dir, files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["metadata"]["test_case"], "unit_test")
        self.assertEqual(data["board"][0][0], 1)

if __name__ == "__main__":
    unittest.main()
