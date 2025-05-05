import os
import argparse
from data_collector import DataCollector
from tensorflow.keras.optimizers import Adam
from trainer import GomokuModelTrainer
import json


def find_latest_npz(base_dir):
    games_dir = os.path.join(base_dir, "training_data")
    npz_files = [f for f in os.listdir(games_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError("No NPZ files found in the games directory.")
    npz_files.sort(reverse=True)
    return os.path.join(games_dir, npz_files[0])


def save_training_history(history, output_dir, model_type):
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, f"{model_type}_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    print(f"训练过程已保存至: {history_path}")


def main(args):
    base_dir = args.base_dir
    dataset_file = find_latest_npz(base_dir)

    # 初始化数据收集器和训练器
    collector = DataCollector(base_dir=base_dir)
    trainer = GomokuModelTrainer(board_size=args.board_size, base_dir=base_dir)

    print(f"自动加载最新数据: {dataset_file}")

    # 根据选择的类型训练不同的网络
    if args.model_type == 'policy':
        model, history, _ = trainer.train_policy_network(
            dataset_file=dataset_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            conv_filters=args.conv_filters,
            res_blocks=args.res_blocks,
            learning_rate=args.learning_rate
        )
    elif args.model_type == 'value':
        model, history, _ = trainer.train_value_network(
            dataset_file=dataset_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            conv_filters=args.conv_filters,
            res_blocks=args.res_blocks,
            learning_rate=args.learning_rate
        )
    elif args.model_type == 'policy_value':
        model, history, _ = trainer.train_policy_value_network(
            dataset_file=dataset_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            conv_filters=args.conv_filters,
            res_blocks=args.res_blocks,
            learning_rate=args.learning_rate
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # 保存训练过程
    save_training_history(history, os.path.join(base_dir, "training_histories"), args.model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gomoku AI Training and Conversion")

    parser.add_argument('--base_dir', type=str, default='gomoku_data', help='Base directory for data and models')
    parser.add_argument('--board_size', type=int, default=15, help='Gomoku board size')
    parser.add_argument('--model_type', type=str, choices=['policy', 'value', 'policy_value'], required=True, help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--conv_filters', type=int, default=64, help='Number of convolutional filters')
    parser.add_argument('--res_blocks', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    main(args)
