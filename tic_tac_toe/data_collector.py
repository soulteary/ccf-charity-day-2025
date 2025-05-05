import os
import json
from datetime import datetime
import numpy as np

class DataCollector:
    def __init__(self, base_dir="tic_tac_toe_data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_game(self, game, metadata=None):
        if metadata is None:
            metadata = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.base_dir}/game_{timestamp}.json"

        data = {
            "board": game.board.tolist(),
            "winner": game.winner,
            "metadata": metadata
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
