import argparse
from config import DEFAULT_BOARD_SIZE, DEFAULT_BATCH_SIZE
from environment.gomoku_gpu_env import GomokuGPUEnvironment
from agents.gpu_mcts import GPUMCTS
from data_collection.gpu_data_collector import GPUDataCollector
from common.logger import logger
from tqdm import tqdm
import tensorflow as tf

def main(args):
    # 初始化数据收集器，Gomoku环境和MCTS算法
    logger.info("Initializing the data collector, Gomoku environment, and MCTS algorithm.")
    collector = GPUDataCollector(base_dir=args.base_dir)
    env = GomokuGPUEnvironment(batch_size=args.batch_size, board_size=args.board_size)
    mcts = GPUMCTS(iterations=args.mcts_iterations, board_size=args.board_size)

    games_played = 0
    total_games = args.num_games
    logger.info(f"Starting the training for {total_games} games.")

    # 使用tqdm显示游戏进度
    with tqdm(total=total_games, desc="Total Games Played") as pbar:
        while games_played < total_games:
            # 重置环境，每局新游戏
            logger.info(f"Starting game {games_played + 1}")
            env.reset()
            active_games = tf.ones(args.batch_size, dtype=tf.bool)  # 所有游戏处于激活状态
            game_history = [[] for _ in range(args.batch_size)]  # 初始化每个游戏的历史记录

            # 运行游戏直到所有游戏结束
            while tf.reduce_any(active_games):  # 当有游戏未结束时
                # 确保env.boards和env.current_players为int64类型
                boards = tf.cast(env.boards, tf.int64)
                players = tf.cast(env.current_players, tf.int64)

                # 获取MCTS返回的动作，并确保数据类型为int32
                actions = tf.cast(mcts.search(boards, players), tf.int32)
                env.step(actions)  # 执行动作

                # 更新每个游戏的历史记录
                for idx, active in enumerate(active_games.numpy()):
                    if active:  # 如果该局游戏未结束
                        game_history[idx].append({"move": actions[idx].numpy().tolist(),
                                                  "player": int(env.current_players.numpy()[idx])})
                        # 如果该局游戏结束，将其标记为已完成
                        if env.done.numpy()[idx]:
                            active_games = tf.tensor_scatter_nd_update(active_games, [[idx]], [False])

            # 保存游戏数据
            winners_np = env.winners.numpy()  # 获取所有游戏的赢家
            for idx in range(args.batch_size):
                try:
                    collector.save_game_data(f"game_{games_played + idx}", game_history[idx], int(winners_np[idx]), 
                                             {"mcts_iterations": args.mcts_iterations, "board_size": args.board_size})
                except Exception as e:
                    logger.error(f"Error saving game data for game {games_played + idx}: {e}")

            games_played += args.batch_size  # 更新已玩的游戏数
            pbar.update(args.batch_size)  # 更新进度条
            logger.info(f"Games played: {games_played}/{total_games}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Optimized Gomoku AI Training")
    parser.add_argument('--num_games', type=int, default=5000, help="设置要进行的游戏数量")
    parser.add_argument('--mcts_iterations', type=int, default=5000, help="设置每局游戏MCTS的迭代次数")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="设置每批次的游戏数量")
    parser.add_argument('--board_size', type=int, default=DEFAULT_BOARD_SIZE, help="设置棋盘的大小")
    parser.add_argument('--base_dir', type=str, default='improved_gomoku_data', help="设置数据存储的目录")

    # 获取并传递命令行参数
    args = parser.parse_args()
    logger.info("Starting Gomoku AI training...")
    main(args)
    logger.info("Training completed.")
