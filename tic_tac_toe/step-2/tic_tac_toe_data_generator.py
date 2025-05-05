import numpy as np
import os
import random
import argparse
from datetime import datetime
import json
from tqdm import tqdm
from tic_tac_toe import TicTacToeGame
from agents import RandomAgent, GreedyAgent

class TicTacToeDataGenerator:
    def __init__(self, base_dir="tic_tac_toe_training_data"):
        self.base_dir = base_dir
        self.json_dir = os.path.join(base_dir, "detailed_json")
        self.npz_dir = os.path.join(base_dir, "npz_data")
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.npz_dir, exist_ok=True)

    def generate_self_play_data(self, num_games=1000, agents=None):
        if agents is None:
            agents = [GreedyAgent(), RandomAgent()]

        detailed_games_data = []
        training_data = []

        for game_idx in tqdm(range(num_games), desc="Generating Tic-Tac-Toe data"):
            game = TicTacToeGame()
            game_steps = []

            while not game.game_over:
                current_agent = agents[game.current_player - 1]
                move = current_agent.select_move(game)
                board_before = game.board.copy()
                game.make_move(*move)
                board_after = game.board.copy()

                # 保存每一步详细信息
                game_steps.append({
                    "step": len(game_steps) + 1,
                    "player": int(game.current_player),
                    "move": [int(move[0]), int(move[1])],
                    "board_before": board_before.astype(int).tolist(),
                    "board_after": board_after.astype(int).tolist()
                })

                # 为NPZ训练数据准备
                training_data.append({
                    "state": board_before.copy(),
                    "move": move,
                    "player": game.current_player,
                    "winner": None  # 后续标记
                })

            # 更新胜利信息（标记获胜方）
            winner = game.winner
            for entry in training_data[-len(game_steps):]:
                if winner == 0:
                    entry["winner"] = 0.5  # 平局
                elif winner == entry["player"]:
                    entry["winner"] = 1  # 赢
                else:
                    entry["winner"] = 0  # 输

            detailed_games_data.append({
                "game_id": int(game_idx),
                "winner": int(winner),
                "steps": game_steps
            })

        return detailed_games_data, training_data

    def save_detailed_json(self, detailed_games_data, filename=None):
        if filename is None:
            filename = f"detailed_games_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.json_dir, filename), 'w') as f:
            json.dump(detailed_games_data, f, indent=2)

    def save_npz_data(self, training_data, filename=None, train_ratio=0.8):
        random.shuffle(training_data)

        split_idx = int(len(training_data) * train_ratio)
        train_set = training_data[:split_idx]
        test_set = training_data[split_idx:]

        def format_data(data_set):
            X, y = [], []
            for entry in data_set:
                state_flat = entry["state"].flatten()
                move_idx = entry["move"][0] * 3 + entry["move"][1]
                move_one_hot = np.zeros(9)
                move_one_hot[move_idx] = 1
                X.append(np.concatenate([state_flat, move_one_hot]))
                y.append(entry["winner"])
            return np.array(X), np.array(y)

        X_train, y_train = format_data(train_set)
        X_test, y_test = format_data(test_set)

        if filename is None:
            filename = f"tic_tac_toe_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"

        np.savez_compressed(
            os.path.join(self.npz_dir, filename),
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )

def main():
    parser = argparse.ArgumentParser(description="Generate Tic-Tac-Toe detailed and NPZ training data")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to simulate")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    parser.add_argument("--output_dir", type=str, default="tic_tac_toe_training_data", help="Output base directory")
    parser.add_argument("--agents", type=str, nargs=2, default=["greedy", "random"],
                        choices=["random", "greedy"], help="Agent types")

    args = parser.parse_args()

    agent_dict = {
        "random": RandomAgent(),
        "greedy": GreedyAgent()
    }
    agents = [agent_dict[name] for name in args.agents]

    generator = TicTacToeDataGenerator(base_dir=args.output_dir)
    detailed_data, npz_data = generator.generate_self_play_data(num_games=args.num_games, agents=agents)

    generator.save_detailed_json(detailed_data)
    generator.save_npz_data(npz_data, train_ratio=args.train_ratio)

if __name__ == "__main__":
    main()
