import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

# 设置中文字体支持，避免绘图报错
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_board(board, board_size, last_move=None, save_path=None, show=False):
    plt.figure(figsize=(10, 10))
    plt.gca().set_facecolor('#DEB887')  # 棋盘背景

    for i in range(board_size):
        plt.plot([0, board_size-1], [i, i], 'k', lw=0.5)
        plt.plot([i, i], [0, board_size-1], 'k', lw=0.5)

    center = board_size // 2
    star_positions = []

    if board_size >= 15:
        star_positions = [(center, center), (3, 3), (3, center), (3, board_size-4),
                          (center, 3), (center, board_size-4),
                          (board_size-4, 3), (board_size-4, center), (board_size-4, board_size-4)]
    elif board_size >= 9:
        star_positions = [(center, center), (2, 2), (2, board_size-3),
                          (board_size-3, 2), (board_size-3, board_size-3)]

    for pos in star_positions:
        plt.plot(pos[1], pos[0], 'o', markersize=5, color='black')

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 1:
                plt.plot(j, i, 'o', markersize=18, color='black')
            elif board[i, j] == 2:
                plt.plot(j, i, 'o', markersize=18, color='white', markeredgecolor='black')

    if last_move:
        row, col, player = last_move
        marker_color = 'white' if player == 1 else 'black'
        plt.plot(col, row, 'x', markersize=8, color=marker_color)

    plt.grid(False)
    plt.xticks(range(board_size))
    plt.yticks(range(board_size))
    plt.title("Gomoku Game")
    plt.xlim(-0.5, board_size - 0.5)
    plt.ylim(-0.5, board_size - 0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    if show:
        plt.show()
    else:
        plt.close()


def create_html_viewer(game_id, game_data, image_paths, html_path):
    rel_image_paths = [os.path.relpath(path, os.path.dirname(html_path)).replace("\\", "/") for path in image_paths]

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>五子棋游戏查看器 - {game_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; margin: 20px; }}
            .control-panel {{ margin: 20px; }}
            .board-container {{ margin: 20px auto; }}
            button {{ padding: 5px 10px; margin: 0 5px; }}
            .move-info {{ margin: 10px; }}
        </style>
    </head>
    <body>
        <h1>五子棋游戏查看器</h1>
        <div class="game-info">
            <p><strong>游戏ID:</strong> {game_id}</p>
            <p><strong>棋盘大小:</strong> {game_data["board_size"]}×{game_data["board_size"]}</p>
            <p><strong>结果:</strong> {"黑胜" if game_data["winner"] == 1 else "白胜" if game_data["winner"] == 2 else "平局"}</p>
            <p><strong>总步数:</strong> {len(game_data["move_history"])}</p>
        </div>
        <div class="control-panel">
            <button onclick="firstMove()">第一步</button>
            <button onclick="prevMove()">上一步</button>
            <button onclick="nextMove()">下一步</button>
            <button onclick="lastMove()">最后一步</button>
            <button onclick="playAnimation()">播放</button>
            <button onclick="stopAnimation()">停止</button>
        </div>
        <div class="move-info">
            <span id="moveNumber">当前：第0步（初始棋盘）</span>
        </div>
        <div class="board-container">
            <img id="boardImage" src="{rel_image_paths[0]}" alt="棋盘" style="max-width: 600px;">
        </div>

        <script>
            const imagePaths = {str(rel_image_paths).replace("'", '"')};
            let currentMove = 0;
            let animationInterval = null;

            function updateBoard() {{
                document.getElementById('boardImage').src = imagePaths[currentMove];
                document.getElementById('moveNumber').textContent =
                    '当前：第' + currentMove + '步' + (currentMove === 0 ? '（初始棋盘）' : '');
            }}

            function firstMove() {{ currentMove = 0; updateBoard(); }}
            function prevMove() {{ if (currentMove > 0) {{ currentMove--; updateBoard(); }} }}
            function nextMove() {{ if (currentMove < imagePaths.length - 1) {{ currentMove++; updateBoard(); }} }}
            function lastMove() {{ currentMove = imagePaths.length - 1; updateBoard(); }}

            function playAnimation() {{
                stopAnimation();
                animationInterval = setInterval(() => {{
                    if (currentMove < imagePaths.length - 1) {{
                        currentMove++;
                        updateBoard();
                    }} else {{
                        stopAnimation();
                    }}
                }}, 1000);
            }}

            function stopAnimation() {{
                if (animationInterval) {{
                    clearInterval(animationInterval);
                    animationInterval = null;
                }}
            }}
        </script>
    </body>
    </html>
    """

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path


def process_single_game(args):
    file_path, output_base_dir, generate_html = args

    try:
        with open(file_path, 'r') as f:
            game_data = json.load(f)

        game_id = game_data["game_id"]
        board_size = game_data["board_size"]
        move_history = game_data["move_history"]

        if not move_history:
            print(f"文件 {file_path} 中没有有效的 move_history，跳过")
            return None

        output_dir = os.path.join(output_base_dir, "visualizations", game_id)
        os.makedirs(output_dir, exist_ok=True)

        board_history = [np.zeros((board_size, board_size), dtype=np.int8)]
        current_board = np.zeros((board_size, board_size), dtype=np.int8)

        for move in move_history:
            row, col, player = move
            current_board[row, col] = player
            board_history.append(current_board.copy())

        image_paths = []
        for step, board in enumerate(board_history):
            last_move = None if step == 0 else tuple(move_history[step - 1])
            img_path = os.path.join(output_dir, f"move_{step:03d}.png")
            visualize_board(board, board_size, last_move, save_path=img_path)
            image_paths.append(img_path)

        if generate_html:
            html_path = os.path.join(output_base_dir, "visualizations", f"{game_id}_viewer.html")
            create_html_viewer(game_id, game_data, image_paths, html_path)

        return game_id

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        traceback.print_exc()
        return None


def main(args):
    base_dir = args.data_dir
    games_dir = os.path.join(base_dir, "games")
    output_dir = args.output_dir or base_dir

    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

    game_files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith(".json")]
    print(f"找到 {len(game_files)} 个游戏文件，准备生成可视化...")

    cpu_cores = min(cpu_count(), args.cpu_cores)
    tasks = [(fp, output_dir, args.generate_html) for fp in game_files]

    with Pool(processes=cpu_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_game, tasks), total=len(tasks)):
            pass

    print(f"可视化生成完成。结果保存在: {os.path.join(output_dir, 'visualizations')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="五子棋游戏数据可视化生成工具")
    parser.add_argument('--data_dir', type=str, default='gomoku_data', help="游戏数据目录")
    parser.add_argument('--output_dir', type=str, default=None, help="输出目录（默认与数据目录相同）")
    parser.add_argument('--cpu_cores', type=int, default=cpu_count(), help="使用的CPU核心数")
    parser.add_argument('--generate_html', action='store_true', help="是否生成HTML查看器")

    args = parser.parse_args()
    print("开始生成五子棋游戏可视化...")
    main(args)
    print("所有任务完成。")
