import argparse
import os
import logging
import glob
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from gomoku_game import GomokuGame
from agents import RandomAgent, GreedyAgent, MCTSAgent
from data_collector import DataCollector

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def play_and_save_game(args_tuple):
    game_idx, board_size, use_mcts, mcts_iterations, base_dir, all_params = args_tuple

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
        "result": "Black Win" if game.winner == 1 else ("White Win" if game.winner == 2 else "Draw")
    }

    # 将参数信息作为单独的参数传递给save_game
    game_id, file_path = collector.save_game(game, metadata, parameters=all_params)
    logger.info(f"[Game {game_idx}] saved with ID: {game_id}, result: {metadata['result']}")

    return game.winner

def generate_games(args):
    """生成游戏数据"""
    total_games = args.num_games
    cpu_cores = min(cpu_count(), args.cpu_cores)

    pool = Pool(processes=cpu_cores)

    # 创建可序列化的参数字典，用于保存
    params_dict = {
        'num_games': args.num_games,
        'board_size': args.board_size,
        'use_mcts': args.use_mcts,
        'mcts_iterations': args.mcts_iterations,
        'base_dir': args.base_dir,
        'cpu_cores': args.cpu_cores,
        'log_interval': args.log_interval
    }

    tasks = [
        (i+1, args.board_size, args.use_mcts, args.mcts_iterations, args.base_dir, params_dict)
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

def generate_npz_from_saved_games(args):
    """从已保存的游戏数据生成NPZ训练文件"""
    logger.info("开始从已保存游戏数据生成NPZ训练文件...")

    # 创建数据收集器
    collector = DataCollector(base_dir=args.base_dir)

    # 加载游戏数据
    games_dir = os.path.join(args.base_dir, "games")
    game_files = glob.glob(os.path.join(games_dir, "*.json"))

    logger.info(f"发现 {len(game_files)} 个游戏数据文件")

    # 如果设置了最大游戏数量，则限制处理文件数
    if args.max_games is not None and args.max_games < len(game_files):
        game_files = game_files[:args.max_games]
        logger.info(f"限制处理前 {args.max_games} 个游戏")

    # 收集所有训练数据
    all_training_data = []
    processed_games = 0
    skipped_games = 0

    for game_file in game_files:
        # 加载游戏数据
        game_data = collector.load_game(file_path=game_file)

        if game_data is None:
            logger.warning(f"无法加载游戏数据文件: {game_file}")
            skipped_games += 1
            continue

        # 应用筛选条件
        if args.filter_winner is not None and game_data["winner"] != args.filter_winner:
            if args.verbose:
                logger.debug(f"跳过游戏 {game_data['game_id']} (胜者不匹配: {game_data['winner']})")
            skipped_games += 1
            continue

        if args.filter_agent is not None:
            if "metadata" not in game_data or "agents" not in game_data["metadata"] or \
               args.filter_agent not in game_data["metadata"]["agents"]:
                if args.verbose:
                    logger.debug(f"跳过游戏 {game_data['game_id']} (智能体不匹配)")
                skipped_games += 1
                continue

        if len(game_data["move_history"]) < args.min_moves:
            if args.verbose:
                logger.debug(f"跳过游戏 {game_data['game_id']} (移动次数不足: {len(game_data['move_history'])})")
            skipped_games += 1
            continue

        # 重建游戏对象
        game = GomokuGame(board_size=game_data["board_size"])
        game.winner = game_data["winner"]
        game.move_history = game_data["move_history"]

        # 重建游戏历史记录（用于训练数据提取）
        game.board_history = []
        current_board = np.zeros((game_data["board_size"], game_data["board_size"]))

        for move in game_data["move_history"]:
            row, col, player = move
            board_copy = current_board.copy()
            board_copy[row, col] = player
            game.board_history.append(board_copy)
            current_board = board_copy

        # 提取训练数据
        training_data = collector.extract_training_data(game)

        if training_data:
            all_training_data.extend(training_data)
            processed_games += 1
            if args.verbose and processed_games % 50 == 0:
                logger.info(f"已处理 {processed_games} 个游戏，当前收集了 {len(all_training_data)} 个训练样本")

    logger.info(f"处理完成: 总共 {processed_games} 个游戏，跳过 {skipped_games} 个游戏")
    logger.info(f"收集了 {len(all_training_data)} 个训练样本")

    # 平衡正负样本
    if args.balance_classes and all_training_data:
        positive_samples = [s for s in all_training_data if s["label"] == 1]
        negative_samples = [s for s in all_training_data if s["label"] == -1]

        logger.info(f"原始数据: 正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")

        if len(positive_samples) > len(negative_samples):
            # 随机采样正样本
            np.random.shuffle(positive_samples)
            positive_samples = positive_samples[:len(negative_samples)]
            all_training_data = positive_samples + negative_samples
        elif len(negative_samples) > len(positive_samples):
            # 随机采样负样本
            np.random.shuffle(negative_samples)
            negative_samples = negative_samples[:len(positive_samples)]
            all_training_data = positive_samples + negative_samples

        logger.info(f"平衡后: 正样本: {len([s for s in all_training_data if s['label'] == 1])}, "
                  f"负样本: {len([s for s in all_training_data if s['label'] == -1])}")

    # 保存NPZ文件
    if all_training_data:
        npz_file = collector.save_training_data_to_npz(all_training_data, file_name=args.output_file)
        logger.info(f"训练数据已保存到: {npz_file}")
    else:
        logger.warning("没有可用的训练数据")

def main():
    """主函数，处理命令行参数和调用相应功能"""
    # 创建主解析器
    main_parser = argparse.ArgumentParser(description="五子棋AI数据处理工具")
    subparsers = main_parser.add_subparsers(dest="command", help="可用命令")

    # 添加游戏生成子命令
    generate_parser = subparsers.add_parser("generate", help="生成新的游戏数据")
    generate_parser.add_argument('--num_games', type=int, default=1000, help="设置要进行的游戏总数量")
    generate_parser.add_argument('--mcts_iterations', type=int, default=2000, help="MCTS算法的迭代次数")
    generate_parser.add_argument('--board_size', type=int, default=15, help="棋盘大小")
    generate_parser.add_argument('--base_dir', type=str, default='gomoku_data', help="数据保存目录")
    generate_parser.add_argument('--use_mcts', action='store_true', help="启用MCTS Agent")
    generate_parser.add_argument('--cpu_cores', type=int, default=cpu_count(), help="设置使用CPU核心数")
    generate_parser.add_argument('--log_interval', type=int, default=20, help="设置日志记录频率")

    # 添加NPZ文件生成子命令
    npz_parser = subparsers.add_parser("make_npz", help="从已有游戏数据生成NPZ训练文件")
    npz_parser.add_argument('--base_dir', type=str, default='gomoku_data',
                           help='游戏数据的根目录路径')
    npz_parser.add_argument('--output_file', type=str, default=None,
                           help='输出NPZ文件名（不包含路径），默认使用时间戳生成')
    npz_parser.add_argument('--filter_winner', type=int, choices=[0, 1, 2], default=None,
                           help='根据胜者筛选游戏（0=平局，1=黑棋获胜，2=白棋获胜），默认不筛选')
    npz_parser.add_argument('--filter_agent', type=str, default=None,
                           help='根据智能体类型筛选游戏，默认不筛选')
    npz_parser.add_argument('--min_moves', type=int, default=0,
                           help='最小移动次数，用于筛选有效游戏')
    npz_parser.add_argument('--max_games', type=int, default=None,
                           help='最大游戏数量限制，默认处理所有游戏')
    npz_parser.add_argument('--balance_classes', action='store_true',
                           help='是否平衡正负样本数量')
    npz_parser.add_argument('--verbose', action='store_true',
                           help='显示详细处理信息')

    # 解析命令行参数
    args = main_parser.parse_args()

    # 根据命令执行对应功能
    if args.command == "generate":
        logger.info("开始Gomoku AI训练...")
        generate_games(args)
        logger.info("训练结束。")
    elif args.command == "make_npz":
        generate_npz_from_saved_games(args)
    else:
        main_parser.print_help()

if __name__ == "__main__":
    main()
