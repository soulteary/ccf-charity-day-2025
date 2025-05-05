import numpy as np
import random
import math
from tic_tac_toe import TicTacToeGame

class RandomAgent:
    def select_move(self, game):
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None

class GreedyAgent:
    def select_move(self, game):
        for move in game.get_valid_moves():
            test_game = game.copy()
            test_game.make_move(*move)
            if test_game.game_over and test_game.winner == game.current_player:
                return move
        return RandomAgent().select_move(game)

class MCTSAgent:
    def __init__(self, simulations=2000, exploration_weight=1.4):
        self.simulations = simulations
        self.C = exploration_weight

    class Node:
        def __init__(self, game, parent=None, move=None):
            self.game = game
            self.parent = parent
            self.move = move
            self.visits = 0
            self.wins = 0
            self.children = []
            self.untried_moves = game.get_valid_moves()

        def expand(self):
            move = self.untried_moves.pop()
            new_game = self.game.copy()
            new_game.make_move(*move)
            child_node = MCTSAgent.Node(new_game, self, move)
            self.children.append(child_node)
            return child_node

        def best_child(self, C):
            return max(
                self.children,
                key=lambda node: node.wins / node.visits + C * math.sqrt(math.log(self.visits) / node.visits)
            )

        def update(self, result):
            self.visits += 1
            self.wins += result

    def select_move(self, game):
        # First check if there's an immediate winning move
        for move in game.get_valid_moves():
            test_game = game.copy()
            test_game.make_move(*move)
            if test_game.game_over and test_game.winner == game.current_player:
                return move
                
        # If no immediate win, proceed with MCTS
        root = self.Node(game.copy())

        for _ in range(self.simulations):
            node = root

            # Selection
            while not node.untried_moves and node.children:
                node = node.best_child(self.C)

            # Expansion
            if node.untried_moves:
                node = node.expand()

            # Simulation
            sim_game = node.game.copy()
            while not sim_game.game_over:
                possible_moves = sim_game.get_valid_moves()
                # Check for winning moves first
                winning_move = None
                for move in possible_moves:
                    temp_game = sim_game.copy()
                    temp_game.make_move(*move)
                    if temp_game.game_over and temp_game.winner == sim_game.current_player:
                        winning_move = move
                        break
                # Block opponent's winning moves if no winning move
                if not winning_move:
                    for move in possible_moves:
                        temp_game = sim_game.copy()
                        next_player = sim_game.current_player
                        temp_game.make_move(*move)
                        for next_move in temp_game.get_valid_moves():
                            next_temp_game = temp_game.copy()
                            next_temp_game.make_move(*next_move)
                            if next_temp_game.game_over and next_temp_game.winner == next_player:
                                winning_move = move
                                break
                        if winning_move:
                            break
                
                # Make the selected move
                if winning_move:
                    sim_game.make_move(*winning_move)
                else:
                    sim_game.make_move(*random.choice(possible_moves))

            # Backpropagation
            if sim_game.winner == game.current_player:
                reward = 1
            elif sim_game.winner == 0:  # Draw
                reward = 0.5
            else:
                reward = 0

            while node:
                node.update(reward)
                reward = 1 - reward  # Flip reward for opponent
                node = node.parent

        # Choose the best child based on most visits
        best_node = max(root.children, key=lambda n: n.visits)
        return best_node.move
