import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
from tqdm import tqdm
import argparse
import glob

class GomokuModelTrainer:
    """五子棋模型训练器"""
    
    def __init__(self, board_size=15, base_dir="gomoku_data"):
        self.board_size = board_size
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 设置GPU内存增长
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"找到 {len(physical_devices)} 个GPU设备")
        else:
            print("未找到GPU，将使用CPU训练")
    
    def build_policy_network(self, conv_filters=64, num_res_blocks=5):
        """构建策略网络 - 预测下一步最佳落子位置"""
        # 输入层：3通道 (黑子、白子、空位)
        input_shape = (self.board_size, self.board_size, 3)
        inputs = layers.Input(shape=input_shape)
        
        # 初始卷积层
        x = layers.Conv2D(filters=conv_filters, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 残差块
        for _ in range(num_res_blocks):
            x = self._residual_block(x, conv_filters)
        
        # 策略头 - 输出每个位置的落子概率
        policy_head = layers.Conv2D(filters=2, kernel_size=1, padding='same')(x)
        policy_head = layers.BatchNormalization()(policy_head)
        policy_head = layers.ReLU()(policy_head)
        policy_head = layers.Flatten()(policy_head)
        policy_output = layers.Dense(self.board_size * self.board_size, activation='softmax', name='policy_output')(policy_head)
        
        model = models.Model(inputs=inputs, outputs=policy_output)
        return model
    
    def build_value_network(self, conv_filters=64, num_res_blocks=5):
        """构建价值网络 - 评估当前棋局胜率"""
        # 输入层：3通道 (黑子、白子、空位)
        input_shape = (self.board_size, self.board_size, 3)
        inputs = layers.Input(shape=input_shape)
        
        # 初始卷积层
        x = layers.Conv2D(filters=conv_filters, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 残差块
        for _ in range(num_res_blocks):
            x = self._residual_block(x, conv_filters)
        
        # 价值头 - 估计当前局面的胜率
        value_head = layers.Conv2D(filters=1, kernel_size=1, padding='same')(x)
        value_head = layers.BatchNormalization()(value_head)
        value_head = layers.ReLU()(value_head)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(64, activation='relu')(value_head)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value_head)
        
        model = models.Model(inputs=inputs, outputs=value_output)
        return model
    
    def build_policy_value_network(self, conv_filters=64, num_res_blocks=5):
        """构建策略价值网络 (类似AlphaGo Zero)"""
        # 输入层：3通道 (黑子、白子、空位)
        input_shape = (self.board_size, self.board_size, 3)
        inputs = layers.Input(shape=input_shape)
        
        # 初始卷积层
        x = layers.Conv2D(filters=conv_filters, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 残差块
        for _ in range(num_res_blocks):
            x = self._residual_block(x, conv_filters)
        
        # 策略头 - 输出每个位置的落子概率
        policy_head = layers.Conv2D(filters=2, kernel_size=1, padding='same')(x)
        policy_head = layers.BatchNormalization()(policy_head)
        policy_head = layers.ReLU()(policy_head)
        policy_head = layers.Flatten()(policy_head)
        policy_output = layers.Dense(self.board_size * self.board_size, activation='softmax', name='policy_output')(policy_head)
        
        # 价值头 - 估计当前局面的胜率
        value_head = layers.Conv2D(filters=1, kernel_size=1, padding='same')(x)
        value_head = layers.BatchNormalization()(value_head)
        value_head = layers.ReLU()(value_head)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(64, activation='relu')(value_head)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value_head)
        
        model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
        return model
    
    def _residual_block(self, x, filters):
        """残差块"""
        shortcut = x
        
        # 第一个卷积层
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # 第二个卷积层
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # 跳跃连接
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        
        return x
    
    def prepare_inputs_from_boards(self, boards):
        """将棋盘状态转换为模型输入格式"""
        # 棋盘表示为3通道：黑子、白子、空位
        inputs = np.zeros((len(boards), self.board_size, self.board_size, 3), dtype=np.float32)
        
        for i, board in enumerate(boards):
            inputs[i, :, :, 0] = (board == 1).astype(np.float32)  # 黑子
            inputs[i, :, :, 1] = (board == 2).astype(np.float32)  # 白子
            inputs[i, :, :, 2] = (board == 0).astype(np.float32)  # 空位
        
        return inputs
    
    def prepare_targets_from_moves(self, moves):
        """将移动转换为策略目标（one-hot编码）"""
        targets = np.zeros((len(moves), self.board_size * self.board_size), dtype=np.float32)
        
        for i, move in enumerate(moves):
            targets[i, move] = 1.0  # one-hot编码
        
        return targets
    
    def train_policy_network(self, dataset_file, epochs=100, batch_size=64, 
                            conv_filters=64, res_blocks=5, learning_rate=0.001):
        """训练策略网络"""
        print("正在加载数据...")
        data = np.load(dataset_file)
        
        # 准备训练数据
        train_boards = data['train_boards']
        train_moves = data['train_moves']
        train_inputs = self.prepare_inputs_from_boards(train_boards)
        train_targets = self.prepare_targets_from_moves(train_moves)
        
        # 准备验证数据
        val_boards = data['val_boards']
        val_moves = data['val_moves']
        val_inputs = self.prepare_inputs_from_boards(val_boards)
        val_targets = self.prepare_targets_from_moves(val_moves)
        
        print(f"训练样本: {len(train_boards)}, 验证样本: {len(val_boards)}")
        
        # 构建模型
        print("构建策略网络...")
        model = self.build_policy_network(conv_filters=conv_filters, num_res_blocks=res_blocks)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型摘要
        model.summary()
        
        # 设置回调
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"policy_model_{timestamp}.h5")
        log_dir = os.path.join(self.logs_dir, f"policy_{timestamp}")
        
        callbacks = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(patience=20, monitor='val_accuracy', restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        # 训练模型
        print("开始训练策略网络...")
        history = model.fit(
            train_inputs, train_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, val_targets),
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存训练配置和结果
        config = {
            "model_type": "policy",
            "timestamp": timestamp,
            "board_size": self.board_size,
            "conv_filters": conv_filters,
            "res_blocks": res_blocks,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_samples": len(train_boards),
            "val_samples": len(val_boards),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "best_val_accuracy": float(max(history.history['val_accuracy'])),
            "model_path": model_path
        }
        
        # 保存训练配置
        config_path = os.path.join(self.models_dir, f"policy_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # 绘制训练曲线
        self._plot_training_history(history, f"策略网络训练 - {timestamp}", 
                                   os.path.join(self.models_dir, f"policy_history_{timestamp}.png"))
        
        print(f"策略网络训练完成，模型保存至: {model_path}")
        return model, history, config
    
    def train_value_network(self, dataset_file, epochs=100, batch_size=64,
                           conv_filters=64, res_blocks=5, learning_rate=0.001):
        """训练价值网络"""
        print("正在加载数据...")
        data = np.load(dataset_file)
        
        # 准备训练数据
        train_boards = data['train_boards']
        train_labels = data['train_labels']
        train_inputs = self.prepare_inputs_from_boards(train_boards)
        
        # 准备验证数据
        val_boards = data['val_boards']
        val_labels = data['val_labels']
        val_inputs = self.prepare_inputs_from_boards(val_boards)
        
        print(f"训练样本: {len(train_boards)}, 验证样本: {len(val_boards)}")
        
        # 构建模型
        print("构建价值网络...")
        model = self.build_value_network(conv_filters=conv_filters, num_res_blocks=res_blocks)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # 打印模型摘要
        model.summary()
        
        # 设置回调
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"value_model_{timestamp}.h5")
        log_dir = os.path.join(self.logs_dir, f"value_{timestamp}")
        
        callbacks = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        # 训练模型
        print("开始训练价值网络...")
        history = model.fit(
            train_inputs, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_inputs, val_labels),
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存训练配置和结果
        config = {
            "model_type": "value",
            "timestamp": timestamp,
            "board_size": self.board_size,
            "conv_filters": conv_filters,
            "res_blocks": res_blocks,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_samples": len(train_boards),
            "val_samples": len(val_boards),
            "final_train_mae": float(history.history['mae'][-1]),
            "final_val_mae": float(history.history['val_mae'][-1]),
            "best_val_loss": float(min(history.history['val_loss'])),
            "model_path": model_path
        }
        
        # 保存训练配置
        config_path = os.path.join(self.models_dir, f"value_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # 绘制训练曲线
        self._plot_training_history(history, f"价值网络训练 - {timestamp}", 
                                   os.path.join(self.models_dir, f"value_history_{timestamp}.png"),
                                   metrics=['loss', 'val_loss', 'mae', 'val_mae'])
        
        print(f"价值网络训练完成，模型保存至: {model_path}")
        return model, history, config
    
    def train_policy_value_network(self, dataset_file, epochs=100, batch_size=64,
                                 conv_filters=64, res_blocks=5, learning_rate=0.001):
        """训练策略价值网络（双头网络）"""
        print("正在加载数据...")
        data = np.load(dataset_file)
        
        # 准备训练数据
        train_boards = data['train_boards']
        train_moves = data['train_moves']
        train_labels = data['train_labels']
        train_inputs = self.prepare_inputs_from_boards(train_boards)
        train_policy_targets = self.prepare_targets_from_moves(train_moves)
        
        # 准备验证数据
        val_boards = data['val_boards']
        val_moves = data['val_moves']
        val_labels = data['val_labels']
        val_inputs = self.prepare_inputs_from_boards(val_boards)
        val_policy_targets = self.prepare_targets_from_moves(val_moves)
        
        print(f"训练样本: {len(train_boards)}, 验证样本: {len(val_boards)}")
        
        # 构建模型
        print("构建策略价值网络...")
        model = self.build_policy_value_network(conv_filters=conv_filters, num_res_blocks=res_blocks)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            },
            metrics={
                'policy_output': ['accuracy'],
                'value_output': ['mae']
            },
            loss_weights={
                'policy_output': 1.0,
                'value_output': 1.0
            }
        )
        
        # 打印模型摘要
        model.summary()
        
        # 设置回调
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"policy_value_model_{timestamp}.h5")
        log_dir = os.path.join(self.logs_dir, f"policy_value_{timestamp}")
        
        callbacks = [
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=20, monitor='val_loss', restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        # 训练模型
        print("开始训练策略价值网络...")
        history = model.fit(
            train_inputs,
            {'policy_output': train_policy_targets, 'value_output': train_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                val_inputs,
                {'policy_output': val_policy_targets, 'value_output': val_labels}
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存训练配置和结果
        config = {
            "model_type": "policy_value",
            "timestamp": timestamp,
            "board_size": self.board_size,
            "conv_filters": conv_filters,
            "res_blocks": res_blocks,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_samples": len(train_boards),
            "val_samples": len(val_boards),
            "final_train_policy_accuracy": float(history.history['policy_output_accuracy'][-1]),
            "final_val_policy_accuracy": float(history.history['val_policy_output_accuracy'][-1]),
            "final_train_value_mae": float(history.history['value_output_mae'][-1]),
            "final_val_value_mae": float(history.history['val_value_output_mae'][-1]),
            "best_val_loss": float(min(history.history['val_loss'])),
            "model_path": model_path
        }
        
        # 保存训练配置
        config_path = os.path.join(self.models_dir, f"policy_value_config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # 绘制训练曲线
        self._plot_training_history(history, f"策略价值网络训练 - {timestamp}", 
                                   os.path.join(self.models_dir, f"policy_value_history_{timestamp}.png"),
                                   metrics=['loss', 'val_loss', 'policy_output_accuracy', 
                                           'val_policy_output_accuracy', 'value_output_mae', 
                                           'val_value_output_mae'])
        
        print(f"策略价值网络训练完成，模型保存至: {model_path}")
        return model, history, config
    
    def _plot_training_history(self, history, title, save_path, metrics=None):
        """绘制训练历史曲线"""
        if metrics is None:
            metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        
        plt.figure(figsize=(15, 10))
        
        # 绘制所有指定指标
        for i, metric in enumerate(metrics):
            if metric in history.history:
                plt.subplot(len(metrics)//2 + len(metrics)%2, 2, i+1)
                plt.plot(history.history[metric])
                plt.title(f'{metric}')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_model(self, model_path, dataset_file, model_type='policy'):
        """评估模型性能"""
        print(f"评估模型: {model_path}")
        
        # 加载模型
        model = tf.keras.models.load_model(model_path)
        
        # 加载数据
        data = np.load(dataset_file)
        val_boards = data['val_boards']
        val_inputs = self.prepare_inputs_from_boards(val_boards)
        
        if model_type == 'policy':
            val_moves = data['val_moves']
            val_targets = self.prepare_targets_from_moves(val_moves)
            
            # 评估
            results = model.evaluate(val_inputs, val_targets, verbose=1)
            print(f"验证损失: {results[0]:.4f}, 验证准确率: {results[1]:.4f}")
            
            # 更详细的分析
            predictions = model.predict(val_inputs)
            top3_accuracy = self._calculate_topk_accuracy(predictions, val_moves, k=3)
            top5_accuracy = self._calculate_topk_accuracy(predictions, val_moves, k=5)
            
            print(f"Top-3 准确率: {top3_accuracy:.4f}")
            print(f"Top-5 准确率: {top5_accuracy:.4f}")
            
            return {
                "loss": float(results[0]),
                "accuracy": float(results[1]),
                "top3_accuracy": float(top3_accuracy),
                "top5_accuracy": float(top5_accuracy)
            }
            
        elif model_type == 'value':
            val_labels = data['val_labels']
            
            # 评估
            results = model.evaluate(val_inputs, val_labels, verbose=1)
            print(f"验证损失 (MSE): {results[0]:.4f}, 验证MAE: {results[1]:.4f}")
            
            # 预测分析
            predictions = model.predict(val_inputs).flatten()
            
            # 计算各种指标
            mse = np.mean((predictions - val_labels) ** 2)
            mae = np.mean(np.abs(predictions - val_labels))
            correlation = np.corrcoef(predictions, val_labels)[0, 1]
            
            print(f"MSE: {mse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"相关系数: {correlation:.4f}")
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "correlation": float(correlation)
            }
            
        elif model_type == 'policy_value':
            val_moves = data['val_moves']
            val_labels = data['val_labels']
            val_policy_targets = self.prepare_targets_from_moves(val_moves)
            
            # 评估
            results = model.evaluate(
                val_inputs,
                {'policy_output': val_policy_targets, 'value_output': val_labels},
                verbose=1
            )
            
            print(f"验证总损失: {results[0]:.4f}")
            print(f"策略损失: {results[1]:.4f}, 策略准确率: {results[3]:.4f}")
            print(f"价值损失: {results[2]:.4f}, 价值MAE: {results[4]:.4f}")
            
            # 更详细的分析
            predictions = model.predict(val_inputs)
            policy_predictions = predictions[0]
            value_predictions = predictions[1].flatten()
            
            # 策略分析
            top3_accuracy = self._calculate_topk_accuracy(policy_predictions, val_moves, k=3)
            top5_accuracy = self._calculate_topk_accuracy(policy_predictions, val_moves, k=5)
            
            print(f"策略 Top-3 准确率: {top3_accuracy:.4f}")
            print(f"策略 Top-5 准确率: {top5_accuracy:.4f}")
            
            # 价值分析
            value_mse = np.mean((value_predictions - val_labels) ** 2)
            value_mae = np.mean(np.abs(value_predictions - val_labels))
            value_correlation = np.corrcoef(value_predictions, val_labels)[0, 1]
            
            print(f"价值 MSE: {value_mse:.4f}")
            print(f"价值 MAE: {value_mae:.4f}")
            print(f"价值相关系数: {value_correlation:.4f}")
            
            return {
                "total_loss": float(results[0]),
                "policy_loss": float(results[1]),
                "policy_accuracy": float(results[3]),
                "policy_top3_accuracy": float(top3_accuracy),
                "policy_top5_accuracy": float(top5_accuracy),
                "value_loss": float(results[2]),
                "value_mae": float(results[4]),
                "value_mse": float(value_mse),
                "value_correlation": float(value_correlation)
            }
    
    def _calculate_topk_accuracy(self, predictions, true_moves, k=3):
        """计算Top-K准确率"""
        correct = 0
        for i, pred in enumerate(predictions):
            # 获取前k个预测
            topk_indices = np.argsort(pred)[-k:]
            if true_moves[i] in topk_indices:
                correct += 1
        
        return correct / len(predictions)
    
    def convert_to_tflite(self, model_path, output_path=None):
        """将模型转换为TensorFlow Lite格式"""
        print(f"正在转换模型: {model_path}")
        
        # 如果未指定输出路径，默认在同一目录
        if output_path is None:
            output_path = model_path.replace('.h5', '.tflite')
        
        # 加载模型
        model = tf.keras.models.load_model(model_path)
        
        # 转换到TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # 保存模型
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite模型保存至: {output_path}")
        return output_path
    
    def convert_to_tfjs(self, model_path, output_dir=None):
        """将模型转换为TensorFlow.js格式"""
        try:
            import tensorflowjs as tfjs
        except ImportError:
            print("未安装tensorflowjs。请使用pip install tensorflowjs安装")
            return None
        
        print(f"正在转换模型: {model_path}")
        
        # 如果未指定输出目录，默认在同一目录
        if output_dir is None:
            model_name = os.path.basename(model_path).replace('.h5', '')
            output_dir = os.path.join(os.path.dirname(model_path), f"tfjs_{model_name}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换模型
        tfjs.converters.save_keras_model(tf.keras.models.load_model(model_path), output_dir)
        
        print(f"TensorFlow.js模型保存至: {output_dir}")
        return output_dir


def find_latest_dataset(base_dir="gomoku_data"):
    """查找最新的数据集文件"""
    dataset_pattern = os.path.join(base_dir, "models", "training_dataset_*.npz")
    dataset_files = glob.glob(dataset_pattern)
    
    if not dataset_files:
        return None
    
    # 按时间戳排序，返回最新的
    latest_file = max(dataset_files, key=os.path.getmtime)
    return latest_file

def find_all_datasets(base_dir="gomoku_data"):
    """查找所有数据集文件"""
    dataset_pattern = os.path.join(base_dir, "models", "training_dataset_*.npz")
    dataset_files = glob.glob(dataset_pattern)
    
    # 按时间戳排序
    dataset_files.sort(key=os.path.getmtime)
    return dataset_files

def find_latest_model(base_dir="gomoku_data", model_type="policy"):
    """查找指定类型的最新模型"""
    model_pattern = os.path.join(base_dir, "models", f"{model_type}_model_*.h5")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        return None
    
    # 按时间戳排序，返回最新的
    latest_file = max(model_files, key=os.path.getmtime)
    return latest_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="五子棋AI模型训练工具")
    
    # 基本参数
    parser.add_argument("--action", type=str, required=True, 
                        choices=["train", "evaluate", "convert"],
                        help="执行的操作：训练、评估或转换模型")
    parser.add_argument("--model_type", type=str, default="policy_value",
                        choices=["policy", "value", "policy_value"],
                        help="模型类型：策略网络、价值网络或双头网络")
    parser.add_argument("--data_dir", type=str, default="gomoku_data",
                        help="数据目录路径")
    
    # 训练参数
    parser.add_argument("--dataset", type=str, default=None,
                        help="训练数据集路径，默认使用最新的")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--filters", type=int, default=64,
                        help="卷积层滤波器数量")
    parser.add_argument("--res_blocks", type=int, default=5,
                        help="残差块数量")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    
    # 评估参数
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径，默认使用最新的")
    
    # 转换参数
    parser.add_argument("--convert_to", type=str, default="tflite",
                        choices=["tflite", "tfjs", "both"],
                        help="转换格式：TensorFlow Lite或TensorFlow.js")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = GomokuModelTrainer(base_dir=args.data_dir)
    
    if args.action == "train":
        # 训练模型
        dataset_file = args.dataset
        if dataset_file is None:
            dataset_file = find_latest_dataset(args.data_dir)
            if dataset_file is None:
                print("未找到数据集文件，请使用--dataset指定或先生成数据集")
                return
        
        print(f"使用数据集: {dataset_file}")
        
        if args.model_type == "policy":
            trainer.train_policy_network(
                dataset_file=dataset_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                conv_filters=args.filters,
                res_blocks=args.res_blocks,
                learning_rate=args.lr
            )
        elif args.model_type == "value":
            trainer.train_value_network(
                dataset_file=dataset_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                conv_filters=args.filters,
                res_blocks=args.res_blocks,
                learning_rate=args.lr
            )
        elif args.model_type == "policy_value":
            trainer.train_policy_value_network(
                dataset_file=dataset_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                conv_filters=args.filters,
                res_blocks=args.res_blocks,
                learning_rate=args.lr
            )
    
    elif args.action == "evaluate":
        # 评估模型
        model_file = args.model
        if model_file is None:
            model_file = find_latest_model(args.data_dir, args.model_type)
            if model_file is None:
                print(f"未找到{args.model_type}类型的模型文件，请使用--model指定")
                return
        
        dataset_file = args.dataset
        if dataset_file is None:
            dataset_file = find_latest_dataset(args.data_dir)
            if dataset_file is None:
                print("未找到数据集文件，请使用--dataset指定或先生成数据集")
                return
        
        print(f"使用模型: {model_file}")
        print(f"使用数据集: {dataset_file}")
        
        trainer.evaluate_model(model_file, dataset_file, args.model_type)
    
    elif args.action == "convert":
        # 转换模型
        model_file = args.model
        if model_file is None:
            model_file = find_latest_model(args.data_dir, args.model_type)
            if model_file is None:
                print(f"未找到{args.model_type}类型的模型文件，请使用--model指定")
                return
        
        print(f"使用模型: {model_file}")
        
        if args.convert_to in ["tflite", "both"]:
            trainer.convert_to_tflite(model_file)
        
        if args.convert_to in ["tfjs", "both"]:
            trainer.convert_to_tfjs(model_file)

if __name__ == "__main__":
    main()
