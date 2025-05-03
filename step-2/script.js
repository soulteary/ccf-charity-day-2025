document.addEventListener('DOMContentLoaded', function() {
    // 游戏常量
    const BOARD_SIZE = 15; // 15x15 的棋盘
    const CELL_SIZE = 40;  // 每个格子的大小
    const PIECE_RADIUS = CELL_SIZE / 2 - 2;
    const BOARD_MARGIN = CELL_SIZE / 2;
    const CANVAS_SIZE = BOARD_SIZE * CELL_SIZE + CELL_SIZE;
    
    // 游戏状态
    let board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));
    let currentPlayer = 1; // 1 为黑子, 2 为白子
    let gameOver = false;
    let moveHistory = [];
    
    // 获取 DOM 元素
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const currentPlayerElement = document.getElementById('currentPlayer');
    const resultMessageElement = document.getElementById('resultMessage');
    const restartButton = document.getElementById('restartButton');
    const undoButton = document.getElementById('undoButton');
    
    // 初始化游戏
    function initGame() {
        // 调整画布大小
        canvas.width = CANVAS_SIZE;
        canvas.height = CANVAS_SIZE;
        
        // 重置游戏状态
        board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0));
        currentPlayer = 1;
        gameOver = false;
        moveHistory = [];
        
        // 更新 UI
        updateCurrentPlayerDisplay();
        resultMessageElement.textContent = '';
        
        // 绘制棋盘
        drawBoard();
    }
    
    // 绘制棋盘
    function drawBoard() {
        // 清空画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 绘制背景
        ctx.fillStyle = '#DEB887';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // 绘制网格线
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        
        for (let i = 0; i < BOARD_SIZE; i++) {
            // 水平线
            ctx.beginPath();
            ctx.moveTo(BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE);
            ctx.lineTo(BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_MARGIN + i * CELL_SIZE);
            ctx.stroke();
            
            // 垂直线
            ctx.beginPath();
            ctx.moveTo(BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN);
            ctx.lineTo(BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN + (BOARD_SIZE - 1) * CELL_SIZE);
            ctx.stroke();
        }
        
        // 标记中心点和星位
        const starPoints = [
            {x: 3, y: 3}, {x: 3, y: 11}, {x: 7, y: 7}, 
            {x: 11, y: 3}, {x: 11, y: 11}
        ];
        
        ctx.fillStyle = '#000';
        starPoints.forEach(point => {
            ctx.beginPath();
            ctx.arc(
                BOARD_MARGIN + point.x * CELL_SIZE, 
                BOARD_MARGIN + point.y * CELL_SIZE, 
                3, 0, Math.PI * 2
            );
            ctx.fill();
        });
        
        // 绘制棋子
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] !== 0) {
                    drawPiece(row, col, board[row][col]);
                }
            }
        }
    }
    
    // 绘制棋子
    function drawPiece(row, col, player) {
        const x = BOARD_MARGIN + col * CELL_SIZE;
        const y = BOARD_MARGIN + row * CELL_SIZE;
        
        ctx.beginPath();
        ctx.arc(x, y, PIECE_RADIUS, 0, Math.PI * 2);
        
        // 创建渐变效果使棋子看起来有立体感
        const gradient = ctx.createRadialGradient(
            x - PIECE_RADIUS / 3, y - PIECE_RADIUS / 3, PIECE_RADIUS / 10,
            x, y, PIECE_RADIUS
        );
        
        if (player === 1) { // 黑子
            gradient.addColorStop(0, '#666');
            gradient.addColorStop(1, '#000');
        } else { // 白子
            gradient.addColorStop(0, '#fff');
            gradient.addColorStop(1, '#ccc');
        }
        
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // 添加边缘
        ctx.strokeStyle = player === 1 ? '#000' : '#888';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // 高亮最后一步棋
        if (moveHistory.length > 0) {
            const lastMove = moveHistory[moveHistory.length - 1];
            if (lastMove.row === row && lastMove.col === col) {
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fillStyle = player === 1 ? '#fff' : '#000';
                ctx.fill();
            }
        }
    }
    
    // 处理点击事件
    function handleClick(event) {
        if (gameOver) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // 计算点击的行列
        let col = Math.round((x - BOARD_MARGIN) / CELL_SIZE);
        let row = Math.round((y - BOARD_MARGIN) / CELL_SIZE);
        
        // 检查是否在有效范围内
        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            // 检查该位置是否已有棋子
            if (board[row][col] === 0) {
                // 记录移动
                moveHistory.push({row, col, player: currentPlayer});
                
                // 放置棋子
                board[row][col] = currentPlayer;
                
                // 绘制更新后的棋盘
                drawBoard();
                
                // 检查是否有赢家
                if (checkWin(row, col, currentPlayer)) {
                    gameOver = true;
                    resultMessageElement.textContent = 
                        currentPlayer === 1 ? '黑子获胜!' : '白子获胜!';
                    resultMessageElement.className = 
                        'result-message ' + (currentPlayer === 1 ? 'black' : 'white');
                } else if (checkDraw()) {
                    gameOver = true;
                    resultMessageElement.textContent = '平局!';
                    resultMessageElement.className = 'result-message';
                } else {
                    // 切换玩家
                    currentPlayer = currentPlayer === 1 ? 2 : 1;
                    updateCurrentPlayerDisplay();
                }
            }
        }
    }
    
    // 检查是否有人赢了
    function checkWin(row, col, player) {
        // 检查的方向：水平、垂直、对角线、反对角线
        const directions = [
            [{row: 0, col: 1}, {row: 0, col: -1}],  // 水平
            [{row: 1, col: 0}, {row: -1, col: 0}],  // 垂直
            [{row: 1, col: 1}, {row: -1, col: -1}], // 对角线
            [{row: 1, col: -1}, {row: -1, col: 1}]  // 反对角线
        ];
        
        for (const [dir1, dir2] of directions) {
            let count = 1;
            
            // 检查第一个方向
            count += countConsecutive(row, col, player, dir1.row, dir1.col);
            
            // 检查相反方向
            count += countConsecutive(row, col, player, dir2.row, dir2.col);
            
            // 如果连续的棋子达到5个或更多，则获胜
            if (count >= 5) {
                return true;
            }
        }
        
        return false;
    }
    
    // 计算在特定方向上连续的相同颜色棋子
    function countConsecutive(row, col, player, rowDir, colDir) {
        let count = 0;
        
        // 从当前位置开始，沿着指定方向移动
        let r = row + rowDir;
        let c = col + colDir;
        
        // 继续检查，直到边界或不同颜色的棋子
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === player) {
            count++;
            r += rowDir;
            c += colDir;
        }
        
        return count;
    }
    
    // 检查是否平局（棋盘已满）
    function checkDraw() {
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === 0) {
                    return false; // 仍有空位
                }
            }
        }
        return true; // 棋盘已满
    }
    
    // 更新当前玩家显示
    function updateCurrentPlayerDisplay() {
        currentPlayerElement.textContent = currentPlayer === 1 ? '黑子' : '白子';
        currentPlayerElement.className = currentPlayer === 1 ? 'black' : 'white';
    }
    
    // 悔棋
    function undoMove() {
        if (moveHistory.length === 0 || gameOver) return;
        
        const lastMove = moveHistory.pop();
        board[lastMove.row][lastMove.col] = 0;
        
        // 恢复当前玩家
        currentPlayer = lastMove.player;
        
        // 重置游戏状态
        gameOver = false;
        resultMessageElement.textContent = '';
        
        // 更新 UI
        updateCurrentPlayerDisplay();
        drawBoard();
    }
    
    // 事件监听器
    canvas.addEventListener('click', handleClick);
    restartButton.addEventListener('click', initGame);
    undoButton.addEventListener('click', undoMove);
    
    // 初始化游戏
    initGame();
});