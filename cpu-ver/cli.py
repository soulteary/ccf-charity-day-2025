import argparse
import os
import logging
from multiprocessing import Pool, cpu_count
from gomoku_game import GomokuGame
from agents import RandomAgent, GreedyAgent, MCTSAgent
from data_collector import DataCollector

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def play_and_save_game(args_tuple):
    game_idx, board_size, use_mcts, mcts_iterations, base_dir = args_tuple

    # 每个进程单独创建实例
    collector = DataCollector(base_dir=base_dir)
    agents = [RandomAgent(), GreedyAgent()]
    if use_mcts:
        agents.append(MCTSAgent(iterations=mcts_iterations))

    game = GomokuGame(board_size=board_size)
    agent_idx = 0

    while not game.game_over:
        agent = agents[agent_idx % len(agents)]
        move = agent.select_move(game)
        if move:
            success, message = game.make_move(move[0], move[1])
            if not success:
                logger.error(f"[Game {game_idx}] Move failed: {message}")
                break
        else:
            logger.error(f"[Game {game_idx}] No valid moves available")
            break
        agent_idx += 1

    metadata = {
        "agents": [agent.__class__.__name__ for agent in agents],
        "result": "黑胜" if game.winner == 1 else ("白胜" if game.winner == 2 else "平局")
    }
    game_id, file_path = collector.save_game(game, metadata)
    logger.info(f"[Game {game_idx}] saved with ID: {game_id}, result: {metadata['result']}")

    return game.winner

def main(args):
    total_games = args.num_games
    cpu_cores = min(cpu_count(), args.cpu_cores)

    pool = Pool(processes=cpu_cores)

    tasks = [
        (i+1, args.board_size, args.use_mcts, args.mcts_iterations, args.base_dir)
        for i in range(total_games)
    ]

    results = {"black_wins": 0, "white_wins": 0, "draws": 0}

    logger.info(f"开始生成 {total_games} 局游戏数据...")
    for idx, winner in enumerate(pool.imap_unordered(play_and_save_game, tasks), 1):
        if winner == 1:
            results["black_wins"] += 1
        elif winner == 2:
            results["white_wins"] += 1
        else:
            results["draws"] += 1

        if idx % args.log_interval == 0 or idx == total_games:
            logger.info(f"进度: {idx}/{total_games}, 黑胜: {results['black_wins']}, 白胜: {results['white_wins']}, 平局: {results['draws']}")

    pool.close()
    pool.join()

    logger.info("所有游戏数据生成完毕。")
    logger.info(f"最终结果: 黑胜 {results['black_wins']}, 白胜 {results['white_wins']}, 平局 {results['draws']}。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku AI训练程序（多进程加速）")
    parser.add_argument('--num_games', type=int, default=1000, help="设置要进行的游戏总数量")
    parser.add_argument('--mcts_iterations', type=int, default=2000, help="MCTS算法的迭代次数")
    parser.add_argument('--board_size', type=int, default=15, help="棋盘大小")
    parser.add_argument('--base_dir', type=str, default='gomoku_data', help="数据保存目录")
    parser.add_argument('--use_mcts', action='store_true', help="启用MCTS Agent")
    parser.add_argument('--cpu_cores', type=int, default=cpu_count(), help="设置使用CPU核心数")
    parser.add_argument('--log_interval', type=int, default=20, help="设置日志记录频率")

    args = parser.parse_args()

    logger.info("开始Gomoku AI训练...")
    main(args)
    logger.info("训练结束。")
