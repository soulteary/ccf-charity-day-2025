import random
import numpy as np
from gomoku_game import GomokuGame

class RandomAgent:
    """随机策略Agent"""
    
    def select_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)


class GreedyAgent:
    """贪心策略Agent"""
    
    def select_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        for move in valid_moves:
            row, col = move
            board_copy = game.board.copy()
            board_copy[row, col] = game.current_player
            
            game_copy = GomokuGame(game.board_size)
            game_copy.board = board_copy
            game_copy.current_player = game.current_player
            
            if game_copy.check_win(row, col):
                return move
        
        opponent = 3 - game.current_player
        for move in valid_moves:
            row, col = move
            board_copy = game.board.copy()
            board_copy[row, col] = opponent
            
            game_copy = GomokuGame(game.board_size)
            game_copy.board = board_copy
            game_copy.current_player = opponent
            
            if game_copy.check_win(row, col):
                return move
        
        preferred_moves = []
        for move in valid_moves:
            row, col = move
            has_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    r, c = row + dr, col + dc
                    if 0 <= r < game.board_size and 0 <= c < game.board_size and game.board[r, c] != 0:
                        has_neighbor = True
                        break
                
                if has_neighbor:
                    break
            
            if has_neighbor:
                preferred_moves.append(move)
        
        if preferred_moves:
            return random.choice(preferred_moves)
        
        center = game.board_size // 2
        center_moves = []
        for move in valid_moves:
            row, col = move
            distance = abs(row - center) + abs(col - center)
            if distance <= game.board_size // 2:
                center_moves.append(move)
        
        if center_moves:
            return random.choice(center_moves)
        
        return random.choice(valid_moves)
    
class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  
        self.children = []
        self.wins = 0
        self.visits = 0
        
        temp_game = GomokuGame(board_size=len(game_state['board']))
        temp_game.board = game_state['board'].copy()
        temp_game.current_player = game_state['current_player']
        self.untried_moves = temp_game.get_valid_moves()
        
        self.player = game_state['current_player']

    def select_child(self):
        """选择UCB1值最高的子节点"""
        c = 1.4  
        
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        return max(self.children, key=lambda child: 
                  child.wins / child.visits + 
                  c * np.sqrt(2 * np.log(self.visits) / child.visits))
    
    def expand(self, game):
        """扩展节点"""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        move = random.choice(valid_moves)
        
        game_copy = GomokuGame(game.board_size)
        game_copy.board = game.board.copy()
        game_copy.current_player = game.current_player
        game_copy.make_move(move[0], move[1])
        
        child = MCTSNode(game_copy.get_state(), self, move)
        self.children.append(child)
        return child
    
    def update(self, result):
        """更新节点统计信息"""
        self.visits += 1
        if result == self.player:
            self.wins += 1
        elif result == 0:
            self.wins += 0.5

class MCTSAgent:
    """蒙特卡洛树搜索Agent"""

    def __init__(self, iterations=1000):
        self.iterations = iterations

    def select_move(self, game):
        # First check if we can win immediately
        valid_moves = game.get_valid_moves()
        if len(valid_moves) == 1:
            return valid_moves[0]
            
        # Check for immediate winning move
        for move in valid_moves:
            row, col = move
            board_copy = game.board.copy()
            board_copy[row, col] = game.current_player
            
            game_copy = GomokuGame(game.board_size)
            game_copy.board = board_copy
            game_copy.current_player = game.current_player
            
            if game_copy.check_win(row, col):
                return move
                
        # Check for blocking opponent's winning move
        opponent = 3 - game.current_player
        for move in valid_moves:
            row, col = move
            board_copy = game.board.copy()
            board_copy[row, col] = opponent
            
            game_copy = GomokuGame(game.board_size)
            game_copy.board = board_copy
            game_copy.current_player = opponent
            
            if game_copy.check_win(row, col):
                return move
        
        # If no immediate win or block, use MCTS
        root = MCTSNode(game.get_state())
        
        for _ in range(self.iterations):
            game_copy = GomokuGame(game.board_size)
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            
            node = root
            # Selection phase
            while not game_copy.game_over and node.children:
                node = node.select_child()
                if node.move:
                    game_copy.make_move(node.move[0], node.move[1])
            
            # Expansion phase
            if not game_copy.game_over:
                new_node = node.expand(game_copy)
                if new_node:
                    node = new_node
                    if node.move:
                        game_copy.make_move(node.move[0], node.move[1])
            
            # Simulation phase - use smarter rollout instead of completely random
            while not game_copy.game_over:
                valid_moves = game_copy.get_valid_moves()
                if not valid_moves:
                    break
                    
                # Check for winning move
                current_player = game_copy.current_player
                winning_move = None
                for m in valid_moves:
                    r, c = m
                    board_copy = game_copy.board.copy()
                    board_copy[r, c] = current_player
                    
                    temp_game = GomokuGame(game_copy.board_size)
                    temp_game.board = board_copy
                    temp_game.current_player = current_player
                    
                    if temp_game.check_win(r, c):
                        winning_move = m
                        break
                
                # If winning move found, make it
                if winning_move:
                    move = winning_move
                else:
                    # Otherwise random move
                    move = random.choice(valid_moves)
                    
                game_copy.make_move(move[0], move[1])
            
            # Backpropagation phase
            winner = game_copy.winner
            while node:
                node.update(winner)
                node = node.parent
        
        if not root.children:
            return random.choice(valid_moves)
        
        # Choose move with best win rate and sufficient visits
        best_child = max(root.children, 
                         key=lambda child: (child.wins / max(child.visits, 1)) + 0.1 * np.sqrt(np.log(root.visits) / max(child.visits, 1)))
        return best_child.move