import numpy as np
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from tic_tac_toe import TicTacToeGame
from agents import MCTSAgent, GreedyAgent, RandomAgent
import multiprocessing as mp
from functools import partial
import concurrent.futures

class AdvancedTicTacToeDataGenerator:
    def __init__(self, base_dir="tic_tac_toe_advanced_data"):
        self.base_dir = base_dir
        self.npz_dir = os.path.join(base_dir, "npz_data")
        self.json_dir = os.path.join(base_dir, "json_data")
        os.makedirs(self.npz_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def _generate_single_game(self, game_id, agents, history_length=3):
        """
        Generate data for a single game, to be used with multiprocessing
        
        Args:
            game_id: ID of the game
            agents: List of two agents to play against each other
            history_length: Number of previous board states to include in features
            
        Returns:
            Tuple of (training_data, game_history)
        """
        np.random.seed(game_id)  # Ensure different random numbers for each process
        
        game = TicTacToeGame()
        training_data = []
        moves_history = []
        # Pre-allocate history array for better memory efficiency
        history = np.zeros((history_length, 3, 3, 2), dtype=np.float32)
        
        move_count = 0
        
        while not game.game_over:
            current_agent = agents[game.current_player - 1]
            current_player = game.current_player
            
            # Get current board representation - use advanced indexing for speed
            board_current = np.zeros((3, 3, 2), dtype=np.float32)
            board_current[:, :, 0] = (game.board == 1).astype(np.float32)
            board_current[:, :, 1] = (game.board == 2).astype(np.float32)
            
            # Save the current board state before making a move
            # Convert NumPy array to standard Python list to avoid JSON serialization issues
            board_list = game.board.tolist() if isinstance(game.board, np.ndarray) else game.board
            moves_history.append({
                "board": board_list,  
                "current_player": int(current_player),  # Ensure Python native type
                "move_number": int(move_count)          # Ensure Python native type
            })
            
            # Select move using the appropriate agent
            move = current_agent.select_move(game)
            
            # Update the history window with efficient rolling
            history = np.roll(history, -1, axis=0)
            history[-1] = board_current
            
            # Create input feature for the model - preallocate once outside loop
            input_feature = np.reshape(history, (3, 3, history_length * 2))
            
            # Make the move
            game.make_move(*move)
            move_count += 1
            
            # Record the move outcome
            # Convert the move to Python native list to avoid JSON serialization issues
            move_list = [int(move[0]), int(move[1])] if isinstance(move[0], np.integer) or isinstance(move[1], np.integer) else list(move)
            moves_history[-1]["move_made"] = move_list
            
            # Create label: [win, loss, draw] - use array instead of list for better performance
            if game.game_over:
                if game.winner == current_player:
                    label = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Win
                elif game.winner == 0:
                    label = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Draw
                else:
                    label = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Loss
            else:
                # For non-terminal states, we'll use a temporary neutral label
                label = np.array([0.33, 0.33, 0.33], dtype=np.float32)
            
            training_data.append((input_feature, label, current_player, move_count))
        
        # Record the game outcome - ensuring all values are Python native types
        game_history = {
            "game_id": int(game_id),
            "moves": moves_history,
            "winner": int(game.winner) if hasattr(game.winner, 'item') else game.winner,
            "total_moves": int(move_count),
            "metadata": {
                "player1_agent": agents[0].__class__.__name__,
                "player2_agent": agents[1].__class__.__name__,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        }
        
        return training_data, game_history

    def generate_data(self, num_games=10000, agents=None, history_length=3, n_workers=None):
        """
        Generate game data for training machine learning models using multiprocessing
        
        Args:
            num_games: Number of games to generate
            agents: List of two agents to play against each other
            history_length: Number of previous board states to include in features
            n_workers: Number of worker processes (defaults to CPU count)
            
        Returns:
            List of tuples containing (input_feature, label)
        """
        if agents is None:
            # Default to MCTS vs Greedy for high-quality games
            agents = [MCTSAgent(simulations=1000), GreedyAgent()]
        
        # Determine optimal number of workers if not specified
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        n_workers = min(n_workers, num_games, 32)  # Limit max workers - too many can cause overhead
        
        print(f"Generating {num_games} games using {n_workers} CPU workers")
        
        # Prepare arguments for multiprocessing
        generate_game_partial = partial(self._generate_single_game, 
                                       agents=agents, 
                                       history_length=history_length)
        
        all_training_data = []
        all_game_histories = []
        
        # Process games in batches to reduce memory usage
        batch_size = min(1000, max(100, num_games // 10))  # Adjust batch size based on total games
        
        try:
            for batch_start in range(0, num_games, batch_size):
                batch_end = min(batch_start + batch_size, num_games)
                batch_size_actual = batch_end - batch_start
                batch_training_data = []
                batch_game_histories = []
                
                with tqdm(total=batch_size_actual, desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                    # Use concurrent.futures for better exception handling and progress tracking
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                        # Submit all jobs
                        future_to_game_id = {
                            executor.submit(generate_game_partial, game_id): game_id 
                            for game_id in range(batch_start, batch_end)
                        }
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(future_to_game_id):
                            game_id = future_to_game_id[future]
                            try:
                                training_data, game_history = future.result()
                                batch_training_data.extend(training_data)
                                batch_game_histories.append(game_history)
                                pbar.update(1)
                            except Exception as e:
                                print(f"Game {game_id} generated an exception: {e}")
                                # Continue with other games
                
                # Extend main data with batch data
                all_training_data.extend(batch_training_data)
                all_game_histories.extend(batch_game_histories)
                
                # Save the batch to disk to free memory
                try:
                    batch_id = batch_start // batch_size + 1
                    self.save_json_data(batch_game_histories, batch_id=batch_id)
                except Exception as e:
                    print(f"Error saving batch {batch_id}: {e}")
                    # Try saving with a simpler approach
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"tic_tac_toe_games_backup_{timestamp}_batch_{batch_id}.json"
                        file_path = os.path.join(self.json_dir, filename)
                        
                        # Convert all NumPy types to Python types before saving
                        safe_histories = []
                        for game in batch_game_histories:
                            # Deep copy with NumPy conversion
                            game_copy = {
                                "game_id": int(game["game_id"]) if isinstance(game["game_id"], np.integer) else game["game_id"],
                                "winner": int(game["winner"]) if isinstance(game["winner"], np.integer) else game["winner"],
                                "total_moves": int(game["total_moves"]) if isinstance(game["total_moves"], np.integer) else game["total_moves"],
                                "metadata": game["metadata"],
                                "moves": []
                            }
                            
                            # Process moves carefully
                            for move in game["moves"]:
                                safe_move = {}
                                for k, v in move.items():
                                    if k == "board" and isinstance(v, np.ndarray):
                                        safe_move[k] = v.tolist()
                                    elif isinstance(v, np.integer):
                                        safe_move[k] = int(v)
                                    elif isinstance(v, np.floating):
                                        safe_move[k] = float(v)
                                    elif isinstance(v, np.ndarray):
                                        safe_move[k] = v.tolist()
                                    else:
                                        safe_move[k] = v
                                game_copy["moves"].append(safe_move)
                                
                            safe_histories.append(game_copy)
                            
                        with open(file_path, 'w') as f:
                            json.dump({"games": safe_histories, "metadata": {"num_games": len(safe_histories)}}, f)
                        print(f"Saved backup JSON data to {file_path}")
                    except Exception as backup_error:
                        print(f"Backup save also failed: {backup_error}")
                
                # Clear batch data to free memory
                batch_training_data = []
                batch_game_histories = []
        
        except Exception as e:
            print(f"Error during data generation: {e}")
            # Continue to post-processing with whatever data was collected
        
        print(f"Collected {len(all_training_data)} training examples from {len(all_game_histories)} games")
        
        # Post-process the training data to update non-terminal state labels
        try:
            processed_data = self.post_process_data(all_training_data)
        except Exception as e:
            print(f"Error during post-processing: {e}")
            # If post-processing fails, return unprocessed data
            processed_data = [(feature, label) for feature, label, _, _ in all_training_data]
        
        return processed_data, all_game_histories

    def post_process_data(self, training_data):
        """
        Post-process the training data to backpropagate terminal state values
        
        Args:
            training_data: List of (features, labels, player, move_count) tuples
            
        Returns:
            Processed list of (features, label) tuples
        """
        print("Post-processing training data...")
        processed_data = []
        
        # Group data by game more efficiently
        game_move_dict = {}
        for item in training_data:
            feature, label, player, move_count = item
            # Ensure player and move_count are hashable Python types, not NumPy types
            player_key = int(player) if isinstance(player, np.integer) else player
            move_count_key = int(move_count // 10) if isinstance(move_count, np.integer) else move_count // 10
            game_key = (player_key, move_count_key)  # Group by player and approximate game
            
            if game_key not in game_move_dict:
                game_move_dict[game_key] = []
            
            game_move_dict[game_key].append((feature, label, player, move_count))
        
        # Use ThreadPoolExecutor for post-processing which is less CPU-intensive
        num_workers = min(mp.cpu_count(), len(game_move_dict))
        print(f"Using {num_workers} threads for post-processing")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for game_moves in game_move_dict.values():
                futures.append(executor.submit(self._process_single_game, game_moves))
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing games"):
                processed_data.extend(future.result())
        
        return processed_data

    def _process_single_game(self, game_moves):
        """Process a single game's moves for backpropagation of rewards"""
        result = []
        
        # Handle empty game_moves gracefully
        if not game_moves:
            return result
            
        try:
            # Sort moves by move count
            game_moves.sort(key=lambda x: x[3])
            
            # Get terminal state labels
            terminal_labels = {}
            for feature, label, player, move_count in game_moves:
                # Convert player to a hashable type if it's a NumPy integer
                player_key = int(player) if isinstance(player, np.integer) else player
                
                # Check if this is a terminal state
                if label[0] > 0.9 or label[1] > 0.9 or label[2] > 0.9:  # More efficient than checking exact equality
                    terminal_labels[player_key] = label
            
            # Use vectorized operations where possible
            discount_factor = 0.9
            for feature, label, player, move_count in game_moves:
                # Convert player and move_count to standard Python types
                player_key = int(player) if isinstance(player, np.integer) else player
                move_count_val = int(move_count) if isinstance(move_count, np.integer) else move_count
                last_move = int(game_moves[-1][3]) if isinstance(game_moves[-1][3], np.integer) else game_moves[-1][3]
                
                # Check if this is not a terminal state
                if not (label[0] > 0.9 or label[1] > 0.9 or label[2] > 0.9):
                    if player_key in terminal_labels:
                        # Apply discount factor based on distance from terminal state
                        steps_from_terminal = max(0, last_move - move_count_val)
                        discount = discount_factor ** steps_from_terminal
                        # Vectorized multiplication
                        new_label = terminal_labels[player_key] * discount
                        result.append((feature, new_label))
                    else:
                        # If no terminal state for this player, use original label
                        result.append((feature, label))
                else:
                    # Use original label for terminal states
                    result.append((feature, label))
        except Exception as e:
            print(f"Error processing game: {e}")
            # In case of error, return the original data instead of failing completely
            for feature, label, _, _ in game_moves:
                result.append((feature, label))
                
        return result

    def save_npz_data(self, training_data, train_ratio=0.8, val_ratio=0.1):
        """
        Save the training data to an NPZ file, split into train/val/test sets
        
        Args:
            training_data: List of (features, label) tuples
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
        """
        print("Saving NPZ data...")
        
        # Use more efficient shuffling with numpy
        indices = np.arange(len(training_data))
        np.random.shuffle(indices)
        
        # Pre-allocate arrays for better performance
        # Determine shapes
        feature_shape = training_data[0][0].shape
        label_shape = training_data[0][1].shape
        
        X = np.empty((len(training_data),) + feature_shape, dtype=np.float32)
        y = np.empty((len(training_data),) + label_shape, dtype=np.float32)
        
        # Extract features and labels
        for i, idx in enumerate(indices):
            X[i] = training_data[idx][0]
            y[i] = training_data[idx][1]
        
        # Split the data
        train_idx = int(len(training_data) * train_ratio)
        val_idx = train_idx + int(len(training_data) * val_ratio)
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_tic_tac_toe_data_{timestamp}.npz"
        
        # Use a temporary file to avoid memory issues with large datasets
        temp_file = os.path.join(self.npz_dir, f"temp_{timestamp}.npz")
        
        metadata = np.array([{
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_shape": feature_shape,
            "timestamp": timestamp
        }], dtype=object)
        
        # Save in chunks to reduce memory usage
        np.savez_compressed(
            temp_file,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metadata=metadata
        )
        
        # Rename the temporary file
        os.rename(temp_file, os.path.join(self.npz_dir, filename))
        
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
        file_path = os.path.join(self.json_dir, filename)
        
        # Create a custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Use a more memory-efficient approach for writing large JSON files
        with open(file_path, 'w') as f:
            # Write the opening
            f.write('{\n  "games": [\n')
            
            # Write each game separately
            for i, game in enumerate(game_histories):
                # Convert NumPy types to Python native types
                game_json = json.dumps(game, indent=2, cls=NumpyEncoder)
                
                # Adjust indentation
                game_json = '    ' + game_json.replace('\n', '\n    ')
                
                # Add comma if not the last game
                if i < len(game_histories) - 1:
                    game_json += ','
                
                f.write(game_json + '\n')
            
            # Write the metadata and closing
            f.write('  ],\n')
            f.write('  "metadata": {\n')
            f.write(f'    "num_games": {len(game_histories)},\n')
            f.write(f'    "timestamp": "{timestamp}"\n')
            f.write('  }\n')
            f.write('}\n')
        
        print(f"Saved JSON data to {file_path}")

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
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: use all available CPUs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
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
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"Generating {args.num_games} games with {args.agent1} vs {args.agent2}")
    print(f"Using history length of {args.history_length}")
    
    # Create the agents
    agent1 = create_agent(args.agent1, args.mcts_simulations)
    agent2 = create_agent(args.agent2, args.mcts_simulations)
    
    generator = AdvancedTicTacToeDataGenerator(base_dir=args.output_dir)
    
    # Generate the data
    start_time = datetime.now()
    training_data, game_histories = generator.generate_data(
        num_games=args.num_games,
        agents=[agent1, agent2],
        history_length=args.history_length,
        n_workers=args.workers
    )
    
    # Save the data
    generator.save_npz_data(training_data)
    
    # Make sure we save any remaining games
    remaining_games = len(game_histories) % args.save_frequency
    if remaining_games > 0:
        batch_id = len(game_histories) // args.save_frequency
        generator.save_json_data(game_histories[-remaining_games:], batch_id=batch_id)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print(f"Data generation completed in {elapsed:.2f} seconds")
    print(f"Average time per game: {elapsed/args.num_games:.4f} seconds")

if __name__ == "__main__":
    main()