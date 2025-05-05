import numpy as np
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from tic_tac_toe import TicTacToeGame
from agents import MCTSAgent, GreedyAgent, RandomAgent

class AdvancedTicTacToeDataGenerator:
    def __init__(self, base_dir="tic_tac_toe_advanced_data"):
        self.base_dir = base_dir
        self.npz_dir = os.path.join(base_dir, "npz_data")
        self.json_dir = os.path.join(base_dir, "json_data")
        os.makedirs(self.npz_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def generate_data(self, num_games=10000, agents=None, history_length=3):
        """
        Generate game data for training machine learning models
        
        Args:
            num_games: Number of games to generate
            agents: List of two agents to play against each other
            history_length: Number of previous board states to include in features
            
        Returns:
            List of tuples containing (input_feature, label)
        """
        if agents is None:
            # Default to MCTS vs Greedy for high-quality games
            agents = [MCTSAgent(simulations=1000), GreedyAgent()]

        training_data = []
        game_histories = []

        for game_id in tqdm(range(num_games), desc="Generating Advanced Data"):
            game = TicTacToeGame()
            # Track board states for the entire game to record as game history
            moves_history = []
            # Track states for the feature history window
            history = [np.zeros((3, 3, 2)) for _ in range(history_length)]
            
            move_count = 0
            
            while not game.game_over:
                current_agent = agents[game.current_player - 1]
                current_player = game.current_player
                
                # Get current board representation
                board_current = np.stack([game.board == 1, game.board == 2], axis=-1).astype(np.float32)
                
                # Save the current board state before making a move
                moves_history.append({
                    "board": np.copy(game.board).tolist(),
                    "current_player": current_player,
                    "move_number": move_count
                })
                
                # Select move using the appropriate agent
                move = current_agent.select_move(game)
                
                # Update the history window
                history.append(board_current)
                if len(history) > history_length:
                    history.pop(0)
                
                # Create input feature for the model
                input_feature = np.concatenate(history, axis=-1)
                
                # Make the move
                game.make_move(*move)
                move_count += 1
                
                # Record the move outcome
                moves_history[-1]["move_made"] = list(move)
                
                # Create label: [win, loss, draw]
                if game.game_over:
                    if game.winner == current_player:
                        label = [1.0, 0.0, 0.0]  # Win
                    elif game.winner == 0:
                        label = [0.0, 0.0, 1.0]  # Draw
                    else:
                        label = [0.0, 1.0, 0.0]  # Loss
                else:
                    # For non-terminal states, we'll use a temporary neutral label
                    # that will be updated in the post-processing phase
                    label = [0.33, 0.33, 0.33]
                
                training_data.append((input_feature, label, current_player, move_count))
            
            # Record the game outcome
            game_histories.append({
                "game_id": game_id,
                "moves": moves_history,
                "winner": game.winner,
                "total_moves": move_count,
                "metadata": {
                    "player1_agent": agents[0].__class__.__name__,
                    "player2_agent": agents[1].__class__.__name__,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
            })
            
            # Every 1000 games, save the accumulated game histories
            if (game_id + 1) % 1000 == 0:
                self.save_json_data(game_histories[-1000:], batch_id=game_id // 1000)
        
        # Post-process the training data to update non-terminal state labels
        processed_data = self.post_process_data(training_data)
        
        return processed_data, game_histories

    def post_process_data(self, training_data):
        """
        Post-process the training data to backpropagate terminal state values
        
        Args:
            training_data: List of (features, labels, player, move_count) tuples
            
        Returns:
            Processed list of (features, label) tuples
        """
        processed_data = []
        
        # Group data by game
        games_data = {}
        for item in training_data:
            feature, label, player, move_count = item
            if move_count not in games_data:
                games_data[move_count] = []
            games_data[move_count].append((feature, label, player))
        
        # Process each game
        for game_moves in games_data.values():
            # Sort moves by player and move count
            game_moves.sort(key=lambda x: (x[2], x[3]))
            
            # Get terminal state labels
            terminal_labels = {}
            for feature, label, player in game_moves:
                if label[0] == 1.0 or label[1] == 1.0 or label[2] == 1.0:
                    terminal_labels[player] = label
            
            # Update non-terminal states with discounted rewards
            discount_factor = 0.9
            for feature, label, player in game_moves:
                if label[0] != 1.0 and label[1] != 1.0 and label[2] != 1.0:
                    if player in terminal_labels:
                        # Apply discount factor based on distance from terminal state
                        steps_from_terminal = max(0, game_moves[-1][3] - move_count)
                        discount = discount_factor ** steps_from_terminal
                        new_label = [val * discount for val in terminal_labels[player]]
                        processed_data.append((feature, new_label))
                    else:
                        # If no terminal state for this player, use original label
                        processed_data.append((feature, label))
                else:
                    # Use original label for terminal states
                    processed_data.append((feature, label))
        
        return processed_data

    def save_npz_data(self, training_data, train_ratio=0.8, val_ratio=0.1):
        """
        Save the training data to an NPZ file, split into train/val/test sets
        
        Args:
            training_data: List of (features, label) tuples
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
        """
        np.random.shuffle(training_data)
        
        # Extract features and labels
        X = np.array([x[0] for x in training_data])
        y = np.array([x[1] for x in training_data])
        
        # Split the data
        train_idx = int(len(training_data) * train_ratio)
        val_idx = train_idx + int(len(training_data) * val_ratio)
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_tic_tac_toe_data_{timestamp}.npz"
        
        np.savez_compressed(
            os.path.join(self.npz_dir, filename),
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metadata=np.array([{
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "feature_shape": X_train[0].shape,
                "timestamp": timestamp
            }], dtype=object)
        )
        
        print(f"Saved NPZ data to {os.path.join(self.npz_dir, filename)}")
        print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples")
        
        return os.path.join(self.npz_dir, filename)

    def save_json_data(self, game_histories, batch_id=None):
        """
        Save the game histories to a JSON file
        
        Args:
            game_histories: List of game history dictionaries
            batch_id: Optional ID for this batch of games
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_str = f"_batch_{batch_id}" if batch_id is not None else ""
        filename = f"tic_tac_toe_games_{timestamp}{batch_str}.json"
        
        with open(os.path.join(self.json_dir, filename), 'w') as f:
            json.dump({
                "games": game_histories,
                "metadata": {
                    "num_games": len(game_histories),
                    "timestamp": timestamp
                }
            }, f, indent=2)
        
        print(f"Saved JSON data to {os.path.join(self.json_dir, filename)}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate advanced Tic-Tac-Toe training data")
    parser.add_argument("--num_games", type=int, default=10000, 
                        help="Number of games to generate")
    parser.add_argument("--history_length", type=int, default=3, 
                        help="Number of previous board states to include in features")
    parser.add_argument("--output_dir", type=str, default="tic_tac_toe_advanced_data", 
                        help="Directory to save the generated data")
    parser.add_argument("--agent1", type=str, default="mcts", 
                        choices=["mcts", "greedy", "random"], help="First agent type")
    parser.add_argument("--agent2", type=str, default="greedy", 
                        choices=["mcts", "greedy", "random"], help="Second agent type")
    parser.add_argument("--mcts_simulations", type=int, default=1000, 
                        help="Number of simulations for MCTS agent")
    parser.add_argument("--save_frequency", type=int, default=1000, 
                        help="Save JSON data every N games")
    return parser.parse_args()

def create_agent(agent_type, mcts_simulations=1000):
    """Create an agent based on the specified type"""
    if agent_type == "mcts":
        return MCTSAgent(simulations=mcts_simulations)
    elif agent_type == "greedy":
        return GreedyAgent()
    elif agent_type == "random":
        return RandomAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Generating {args.num_games} games with {args.agent1} vs {args.agent2}")
    print(f"Using history length of {args.history_length}")
    
    # Create the agents
    agent1 = create_agent(args.agent1, args.mcts_simulations)
    agent2 = create_agent(args.agent2, args.mcts_simulations)
    
    generator = AdvancedTicTacToeDataGenerator(base_dir=args.output_dir)
    
    # Generate the data
    training_data, game_histories = generator.generate_data(
        num_games=args.num_games,
        agents=[agent1, agent2],
        history_length=args.history_length
    )
    
    # Save the data
    generator.save_npz_data(training_data)
    
    # Make sure we save any remaining games
    remaining_games = len(game_histories) % args.save_frequency
    if remaining_games > 0:
        batch_id = len(game_histories) // args.save_frequency
        generator.save_json_data(game_histories[-remaining_games:], batch_id=batch_id)

if __name__ == "__main__":
    main()