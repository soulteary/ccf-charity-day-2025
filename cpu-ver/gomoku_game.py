import numpy as np

class GomokuGame:
    """五子棋游戏逻辑实现"""
    
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.board_history = []  
        self.save_board_state()  
        return self.get_state()

    def save_board_state(self):
        """保存当前棋盘状态"""
        self.board_history.append(self.board.copy())
    
    def get_state(self):
        """获取当前游戏状态"""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'move_history': self.move_history.copy()
        }
    
    def get_valid_moves(self):
        """获取所有合法的移动"""
        if self.game_over:
            return []
        
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    valid_moves.append((i, j))
        
        return valid_moves
    
    def make_move(self, row, col):
        """执行移动"""
        if self.game_over:
            return False, "游戏已结束"
        
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False, "移动超出棋盘范围"
        
        if self.board[row, col] != 0:
            return False, "该位置已有棋子"
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            self.save_board_state()
            return True, f"玩家 {self.current_player} 获胜"
        
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
            self.save_board_state()
            return True, "平局"
        
        self.current_player = 3 - self.current_player
        self.save_board_state()
        return True, "移动成功"
    
    def check_win(self, row, col):
        """检查是否有玩家获胜"""
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],  
            [(1, 0), (-1, 0)],  
            [(1, 1), (-1, -1)], 
            [(1, -1), (-1, 1)]  
        ]
        
        for dir1, dir2 in directions:
            count = 1  
            for i in range(1, 5):
                r, c = row + dir1[0] * i, col + dir1[1] * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            for i in range(1, 5):
                r, c = row + dir2[0] * i, col + dir2[1] * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        return False
