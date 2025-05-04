import unittest
import os
import tempfile
import json
import numpy as np
from generate_visualizations import visualize_board, create_html_viewer, process_single_game

class TestVisualizer(unittest.TestCase):
    def test_visualize_board_creates_image_file(self):
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1
        board[7, 8] = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "board.png")
            visualize_board(board, board_size=15, last_move=(7, 8, 2), save_path=save_path)
            self.assertTrue(os.path.exists(save_path))

    def test_create_html_viewer_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            game_id = "game_test"
            image_paths = [os.path.join(tmpdir, f"move_{i}.png") for i in range(3)]
            for path in image_paths:
                with open(path, 'w') as f:
                    f.write("fake image")
            game_data = {
                "board_size": 15,
                "winner": 1,
                "move_history": [[7, 7, 1], [7, 8, 2]]
            }
            html_path = os.path.join(tmpdir, "viewer.html")
            create_html_viewer(game_id, game_data, image_paths, html_path)
            self.assertTrue(os.path.exists(html_path))

    def test_process_single_game_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            game_id = "test_game"
            game_data = {
                "game_id": game_id,
                "board_size": 15,
                "move_history": [[7, 7, 1], [7, 8, 2]],
                "winner": 1
            }
            game_file = os.path.join(tmpdir, f"{game_id}.json")
            with open(game_file, 'w') as f:
                json.dump(game_data, f)

            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            result = process_single_game((game_file, output_dir, False))
            self.assertEqual(result, game_id)

if __name__ == '__main__':
    unittest.main()
