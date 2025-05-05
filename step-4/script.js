document.addEventListener('DOMContentLoaded', function () {
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
    let isAIEnabled = false; // 是否启用AI
    let aiPlayer = 2; // AI默认使用白子

    // 获取 DOM 元素
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const currentPlayerElement = document.getElementById('currentPlayer');
    const resultMessageElement = document.getElementById('resultMessage');
    const restartButton = document.getElementById('restartButton');
    const undoButton = document.getElementById('undoButton');
    const toggleAIButton = document.getElementById('toggleAIButton');
    const aiDifficultySelect = document.getElementById('aiDifficulty');

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
            { x: 3, y: 3 }, { x: 3, y: 11 }, { x: 7, y: 7 },
            { x: 11, y: 3 }, { x: 11, y: 11 }
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

        // 如果是AI回合，不处理点击
        if (isAIEnabled && currentPlayer === aiPlayer) return;

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
                makeMove(row, col);

                // 如果启用了AI且游戏未结束，则AI下一步棋
                if (isAIEnabled && !gameOver && currentPlayer === aiPlayer) {
                    setTimeout(makeAIMove, 500); // 延迟一下AI的落子，更自然
                }
            }
        }
    }

    // 落子
    function makeMove(row, col) {
        // 记录移动
        moveHistory.push({ row, col, player: currentPlayer });

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

    // AI落子
    // function makeAIMove() {
    //     const difficulty = aiDifficultySelect.value;
    //     let move;

    //     switch (difficulty) {
    //         case 'easy':
    //             move = makeRandomMove();
    //             break;
    //         case 'medium':
    //             // 有75%的概率使用贪心策略，25%随机
    //             move = Math.random() < 0.75 ? makeGreedyMove() : makeRandomMove();
    //             break;
    //         case 'hard':
    //             move = makeGreedyMove();
    //             break;
    //         default:
    //             move = makeGreedyMove();
    //     }

    //     if (move) {
    //         makeMove(move.row, move.col);
    //     }
    // }

    // 随机策略：随机选择一个空位
    function makeRandomMove() {
        const emptyPositions = [];

        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === 0) {
                    emptyPositions.push({ row, col });
                }
            }
        }

        if (emptyPositions.length > 0) {
            return emptyPositions[Math.floor(Math.random() * emptyPositions.length)];
        }

        return null;
    }

    // 贪心策略：评估每个可能的位置并选择最优位置
    function makeGreedyMove() {
        const player = aiPlayer;
        const opponent = player === 1 ? 2 : 1;
        let bestScore = -Infinity;
        let bestMove = null;

        // 如果是第一步，在中心区域随机选择
        if (moveHistory.length === 0) {
            const centerStart = 5;
            const centerEnd = 9;
            const row = centerStart + Math.floor(Math.random() * (centerEnd - centerStart + 1));
            const col = centerStart + Math.floor(Math.random() * (centerEnd - centerStart + 1));
            return { row, col };
        }

        // 如果是第二步，在上一步棋的周围随机选择
        if (moveHistory.length === 1) {
            const lastMove = moveHistory[0];
            const possibleMoves = [];

            for (let r = -2; r <= 2; r++) {
                for (let c = -2; c <= 2; c++) {
                    if (r === 0 && c === 0) continue;

                    const newRow = lastMove.row + r;
                    const newCol = lastMove.col + c;

                    if (newRow >= 0 && newRow < BOARD_SIZE &&
                        newCol >= 0 && newCol < BOARD_SIZE &&
                        board[newRow][newCol] === 0) {
                        possibleMoves.push({ row: newRow, col: newCol });
                    }
                }
            }

            if (possibleMoves.length > 0) {
                return possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
            }
        }

        // 只评估距离已有棋子3步以内的空位，提高性能
        const candidates = [];
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === 0 && isNearExistingPieces(row, col, 3)) {
                    candidates.push({ row, col });
                }
            }
        }

        // 如果没有候选位置（理论上不应该发生），则使用所有空位
        if (candidates.length === 0) {
            return makeRandomMove();
        }

        // 评估每个候选位置
        for (const move of candidates) {
            // 先考虑是否能直接获胜
            board[move.row][move.col] = player;
            if (checkWin(move.row, move.col, player)) {
                board[move.row][move.col] = 0;
                return move;
            }
            // 没有直接胜负的情况下，评估该位置的分数
            const score = evaluatePosition(move.row, move.col, player);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        return bestMove;
    }

    // 检查一个位置是否靠近现有的棋子
    function isNearExistingPieces(row, col, distance) {
        for (let r = Math.max(0, row - distance); r <= Math.min(BOARD_SIZE - 1, row + distance); r++) {
            for (let c = Math.max(0, col - distance); c <= Math.min(BOARD_SIZE - 1, col + distance); c++) {
                if (board[r][c] !== 0) {
                    return true;
                }
            }
        }
        return false;
    }

    // 评估一个位置的分数
    function evaluatePosition(row, col, player) {
        const opponent = player === 1 ? 2 : 1;
        let score = 0;

        // 检查的方向：水平、垂直、对角线、反对角线
        const directions = [
            [{ row: 0, col: 1 }, { row: 0, col: -1 }],  // 水平
            [{ row: 1, col: 0 }, { row: -1, col: 0 }],  // 垂直
            [{ row: 1, col: 1 }, { row: -1, col: -1 }], // 对角线
            [{ row: 1, col: -1 }, { row: -1, col: 1 }]  // 反对角线
        ];

        // 评估每个方向
        for (const [dir1, dir2] of directions) {
            // 模拟在该位置放置自己的棋子
            board[row][col] = player;
            const selfPatterns = analyzeLinePattern(row, col, player, dir1.row, dir1.col, dir2.row, dir2.col);
            board[row][col] = 0;

            // 模拟在该位置放置对手的棋子
            board[row][col] = opponent;
            const opponentPatterns = analyzeLinePattern(row, col, opponent, dir1.row, dir1.col, dir2.row, dir2.col);
            board[row][col] = 0;

            // 根据棋型评分
            score += scorePattern(selfPatterns, true);  // 自己的得分
            score += scorePattern(opponentPatterns, false); // 对手的得分(防守价值)
        }

        // 额外考虑中心区域加分
        const centerDistance = Math.abs(row - 7) + Math.abs(col - 7);
        if (centerDistance <= 5) {
            score += (6 - centerDistance) * 2;
        }

        return score;
    }

    // 分析一条线上的棋型
    function analyzeLinePattern(row, col, player, dir1Row, dir1Col, dir2Row, dir2Col) {
        const opponent = player === 1 ? 2 : 1;
        const LINE_LENGTH = 9; // 分析的线长度，保证能覆盖五子
        const center = Math.floor(LINE_LENGTH / 2);
        const line = Array(LINE_LENGTH).fill(-1); // -1表示越界, 0表示空位, 1表示己方, 2表示对方

        // 填充线数组
        line[center] = player; // 中心是当前评估的位置

        // 向第一个方向填充
        for (let i = 1; i <= center; i++) {
            const r = row + i * dir1Row;
            const c = col + i * dir1Col;

            if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                line[center + i] = board[r][c];
            } else {
                line[center + i] = -1; // 越界
                break;
            }
        }

        // 向第二个方向填充
        for (let i = 1; i <= center; i++) {
            const r = row + i * dir2Row;
            const c = col + i * dir2Col;

            if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
                line[center - i] = board[r][c];
            } else {
                line[center - i] = -1; // 越界
                break;
            }
        }

        // 转换为字符串便于分析
        let pattern = '';
        for (let i = 0; i < LINE_LENGTH; i++) {
            if (line[i] === player) pattern += 'O'; // 己方
            else if (line[i] === 0) pattern += '.'; // 空位
            else if (line[i] === opponent) pattern += 'X'; // 对方
            else pattern += '#'; // 边界
        }

        return pattern;
    }

    // 根据棋型评分
    function scorePattern(pattern, isOffensive) {
        // 修正一个拼写错误
        if (pattern.includes('OOOOO')) return isOffensive ? 100000 : 80000; // 五连（胜利）

        // 活四和双冲四
        if (pattern.includes('.OOOO.')) return isOffensive ? 10000 : 8000; // 活四
        if ((pattern.match(/\.OOOO[^.]/g) || []).length >= 2 ||
            (pattern.match(/[^.]OOOO\./g) || []).length >= 2) {
            return isOffensive ? 10000 : 8000; // 双冲四
        }

        // 冲四活三
        if ((pattern.includes('.OOO.O') || pattern.includes('O.OOO.')) &&
            pattern.includes('.OOO.')) {
            return isOffensive ? 5000 : 4000;
        }

        // 活三
        if (pattern.includes('.OOO.')) return isOffensive ? 1000 : 800;

        // 眠三
        if (pattern.includes('XOOO.') || pattern.includes('.OOOX') ||
            pattern.includes('OO.O') || pattern.includes('O.OO')) {
            return isOffensive ? 100 : 80;
        }

        // 活二
        if (pattern.includes('..OO..')) return isOffensive ? 50 : 40;

        // 眠二
        if (pattern.includes('.OO.')) return isOffensive ? 10 : 8;

        // 单子
        if (pattern.includes('.O.')) return isOffensive ? 1 : 1;

        return 0;
    }

    // 检查是否有人赢了
    function checkWin(row, col, player) {
        // 检查的方向：水平、垂直、对角线、反对角线
        const directions = [
            [{ row: 0, col: 1 }, { row: 0, col: -1 }],  // 水平
            [{ row: 1, col: 0 }, { row: -1, col: 0 }],  // 垂直
            [{ row: 1, col: 1 }, { row: -1, col: -1 }], // 对角线
            [{ row: 1, col: -1 }, { row: -1, col: 1 }]  // 反对角线
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
        let playerText = currentPlayer === 1 ? '黑子' : '白子';
        if (isAIEnabled && currentPlayer === aiPlayer) {
            playerText += ' (AI)';
        }
        currentPlayerElement.textContent = playerText;
        currentPlayerElement.className = currentPlayer === 1 ? 'black' : 'white';
    }

    // 悔棋
    function undoMove() {
        if (moveHistory.length === 0 || gameOver) return;

        // 如果对战AI，需要撤销两步（玩家和AI的步骤）
        let stepsToUndo = isAIEnabled ? 2 : 1;
        stepsToUndo = Math.min(stepsToUndo, moveHistory.length);

        for (let i = 0; i < stepsToUndo; i++) {
            const lastMove = moveHistory.pop();
            board[lastMove.row][lastMove.col] = 0;
            currentPlayer = lastMove.player;
        }

        // 确保轮到玩家
        if (isAIEnabled && currentPlayer === aiPlayer) {
            currentPlayer = currentPlayer === 1 ? 2 : 1;
        }

        // 重置游戏状态
        gameOver = false;
        resultMessageElement.textContent = '';

        // 更新 UI
        updateCurrentPlayerDisplay();
        drawBoard();
    }

    // 切换AI状态
    function toggleAI() {
        isAIEnabled = !isAIEnabled;
        toggleAIButton.textContent = isAIEnabled ? '关闭AI' : '开启AI';
        aiDifficultySelect.disabled = !isAIEnabled;

        // 更新显示
        updateCurrentPlayerDisplay();

        // 如果开启AI且当前轮到AI，则AI立即落子
        if (isAIEnabled && currentPlayer === aiPlayer && !gameOver) {
            setTimeout(makeAIMove, 500);
        }
    }

    // 事件监听器
    canvas.addEventListener('click', handleClick);
    restartButton.addEventListener('click', initGame);
    undoButton.addEventListener('click', undoMove);
    toggleAIButton.addEventListener('click', toggleAI);

    // 初始化游戏
    initGame();

    // 新增函数，用于调用 Python 模型 API 服务获取落子位置
    async function fetchAIMoveFromAPI() {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                board: board,
                current_player: aiPlayer
            })
        });

        const data = await response.json();
        return data;  // {row: number, col: number}
    }

    // 修改原有的 makeAIMove 函数为：
    async function makeAIMove() {
        const move = await fetchAIMoveFromAPI();
        if (move && !move.error) {
            makeMove(move.row, move.col);
        } else {
            console.error("AI 无法找到有效的移动位置:", move.error);
        }
    }

});