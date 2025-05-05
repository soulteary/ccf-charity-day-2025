import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling2D,
    Reshape, Concatenate
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse
import os
import json
from glob import glob
from datetime import datetime
import time

# 处理 sklearn 依赖问题
try:
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告: sklearn 或 seaborn 未安装，将使用简化版评估。")
    print("可以通过运行 'pip install scikit-learn seaborn' 来安装这些库。")
    SKLEARN_AVAILABLE = False

def find_latest_npz(data_dir):
    """查找指定目录中最新的 NPZ 文件"""
    npz_files = glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"目录 {data_dir} 中未找到 NPZ 文件")
    return max(npz_files, key=os.path.getmtime)

def load_data(npz_file):
    """从 NPZ 文件加载数据，支持验证集划分"""
    try:
        data = np.load(npz_file, allow_pickle=True)

        # 检查文件是否包含验证集
        if "X_val" in data and "y_val" in data:
            return (
                data["X_train"], data["y_train"], 
                data["X_val"], data["y_val"],
                data["X_test"], data["y_test"]
            )
        else:
            # 向后兼容旧数据文件
            print("⚠️ 数据文件中未找到验证集。将使用测试集作为验证集。")
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            # 如果没有验证集，则从训练集中分割出一部分作为验证集
            val_size = int(len(X_train) * 0.1)  # 使用 10% 的训练数据作为验证集
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]

            print(f"已从训练集分割出 {val_size} 个样本作为验证集")

            return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def build_cnn_model(input_shape=(3, 3, 6), dropout_rate=0.3):
    """构建用于井字棋预测的 CNN 模型"""
    inputs = Input(shape=input_shape)

    # 第一个卷积块
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # 第二个卷积块
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 展平和全连接层
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 输出层 (胜、负、平)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_transformer_model(input_shape=(3, 3, 6), num_heads=4, ff_dim=128, dropout_rate=0.2):
    """构建基于 Transformer 的井字棋预测模型"""
    inputs = Input(shape=input_shape)

    # 将 3x3 棋盘重塑为序列
    # 第一条路径：展平并使用 transformer 处理
    x1 = Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)

    # 直接使用 Dense 层进行投影，避免复杂的位置编码
    x1 = Dense(ff_dim)(x1)

    # 应用多头注意力
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=ff_dim // num_heads
    )(x1, x1)
    attention_output = Dropout(dropout_rate)(attention_output)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + attention_output)

    # 前馈网络
    ffn_output = Dense(ff_dim * 2, activation='relu')(x1)
    ffn_output = Dense(ff_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + ffn_output)

    # 第二条路径：使用卷积特征
    x2 = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(ff_dim, (2, 2), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)

    # 合并两条路径
    combined = Concatenate()([Flatten()(x1), x2])
    combined = Dense(ff_dim, activation='relu')(combined)
    combined = LayerNormalization(epsilon=1e-6)(combined)
    combined = Dropout(dropout_rate)(combined)

    # 输出层
    outputs = Dense(3, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model, learning_rate=3e-4, mixed_precision=False):
    """编译模型，使用适当的优化器和指标"""
    try:
        if mixed_precision:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("使用混合精度训练")
            except:
                print("警告: 设置混合精度失败，将使用默认精度。")

        optimizer = Adam(learning_rate=learning_rate)

        # 基本指标
        metrics = ['accuracy']

        # 如果可用，添加高级指标
        try:
            metrics.extend([
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ])
        except:
            print("警告: 无法添加高级指标，仅使用准确率。")

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )
        return model
    except Exception as e:
        print(f"编译模型时出错: {str(e)}")
        # 尝试使用基本配置
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

def create_callbacks(output_dir, model_name, patience=15):
    """创建训练回调"""
    # 创建模型目录
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 基本回调
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),  # 使用 .h5 而不是 .keras 以提高兼容性
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]

    # 可选回调，根据可用性添加
    try:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 3,
                min_lr=1e-6,
                verbose=1
            )
        )
    except:
        print("警告: 无法添加 ReduceLROnPlateau 回调。")
    
    try:
        callbacks.append(
            TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1,
                write_graph=True
            )
        )
    except:
        print("警告: 无法添加 TensorBoard 回调。")

    try:
        callbacks.append(
            CSVLogger(
                os.path.join(model_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        )
    except:
        print("警告: 无法添加 CSVLogger 回调。")

    return callbacks

def plot_training_history(history, output_dir, model_name):
    """绘制训练指标并保存图表"""
    try:
        # 创建图表目录
        plots_dir = os.path.join(output_dir, model_name, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 准确率和损失图表
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练')
        plt.plot(history.history['val_accuracy'], label='验证')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练')
        plt.plot(history.history['val_loss'], label='验证')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_loss.png'), dpi=300)
        plt.close()

        # 如果可用，绘制额外指标
        if 'auc' in history.history:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(history.history['auc'], label='训练')
            plt.plot(history.history['val_auc'], label='验证')
            plt.title('AUC')
            plt.xlabel('轮次')
            plt.ylabel('AUC')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 3, 2)
            plt.plot(history.history['precision'], label='训练')
            plt.plot(history.history['val_precision'], label='验证')
            plt.title('精确率')
            plt.xlabel('轮次')
            plt.ylabel('精确率')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.subplot(1, 3, 3)
            plt.plot(history.history['recall'], label='训练')
            plt.plot(history.history['val_recall'], label='验证')
            plt.title('召回率')
            plt.xlabel('轮次')
            plt.ylabel('召回率')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'additional_metrics.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"绘制训练历史时出错: {str(e)}")
        print("将跳过图表生成。")

def evaluate_model(model, X_test, y_test, output_dir, model_name):
    """评估模型性能并保存结果"""
    # 创建评估目录
    eval_dir = os.path.join(output_dir, model_name, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # 评估模型
    print("\n📊 在测试集上评估模型...")
    test_results = model.evaluate(X_test, y_test, verbose=1)

    # 获取预测结果
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    results = {
        'metrics': {name: float(value) for name, value in zip(model.metrics_names, test_results)}
    }

    # 如果 sklearn 可用，生成混淆矩阵和分类报告
    if SKLEARN_AVAILABLE:
        try:
            # 混淆矩阵
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['胜', '负', '平'],
                        yticklabels=['胜', '负', '平'])
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.title('混淆矩阵')
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()

            # 分类报告
            class_report = classification_report(
                y_true_classes, y_pred_classes,
                target_names=['胜', '负', '平'],
                output_dict=True
            )

            # 添加到结果
            results['classification_report'] = class_report
            results['confusion_matrix'] = cm.tolist()
        except Exception as e:
            print(f"生成高级评估指标时出错: {str(e)}")
    else:
        # 简单的混淆矩阵，不使用 sklearn
        cm = np.zeros((3, 3), dtype=int)
        for i, j in zip(y_true_classes, y_pred_classes):
            cm[i, j] += 1

        # 添加到结果
        results['confusion_matrix'] = cm.tolist()

        # 手动计算每个类别的精确率和召回率
        precision = np.zeros(3)
        recall = np.zeros(3)

        for i in range(3):
            # 精确率：正确预测为类别 i 的样本 / 预测为类别 i 的样本总数
            if np.sum(cm[:, i]) > 0:
                precision[i] = cm[i, i] / np.sum(cm[:, i])

            # 召回率：正确预测为类别 i 的样本 / 类别 i 的样本总数
            if np.sum(cm[i, :]) > 0:
                recall[i] = cm[i, i] / np.sum(cm[i, :])

        # 添加到结果
        results['manual_metrics'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }

    # 保存结果
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results

def save_model_summary(model, output_dir, model_name):
    """将模型架构摘要保存到文本文件"""
    try:
        summary_path = os.path.join(output_dir, model_name, 'model_summary.txt')

        # 重定向 stdout 以捕获摘要
        import io
        import sys
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # 打印模型摘要
        model.summary()

        # 恢复 stdout 并保存捕获的输出
        sys.stdout = original_stdout
        with open(summary_path, 'w') as f:
            f.write(captured_output.getvalue())
    except Exception as e:
        print(f"保存模型摘要时出错: {str(e)}")

def build_ensemble_model(input_shape, dropout_rate=0.2):
    """构建 CNN 和 Transformer 的集成模型"""
    try:
        # 创建单独的 CNN 和 Transformer 模型
        cnn_model = build_cnn_model(input_shape=input_shape, dropout_rate=dropout_rate)
        transformer_model = build_transformer_model(
            input_shape=input_shape,
            dropout_rate=dropout_rate
        )

        # 创建集成输入
        inputs = Input(shape=input_shape)
    
        # 获取两个模型的输出
        cnn_output = cnn_model(inputs)
        transformer_output = transformer_model(inputs)
    
        # 合并输出
        combined = Concatenate()([cnn_output, transformer_output])
        combined = Dense(32, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(dropout_rate)(combined)
        outputs = Dense(3, activation='softmax')(combined)

        model = Model(inputs=inputs, outputs=outputs)
        return model
    except Exception as e:
        print(f"构建集成模型时出错: {str(e)}")
        print("返回备用 CNN 模型...")
        return build_cnn_model(input_shape=input_shape, dropout_rate=dropout_rate)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练井字棋预测模型')
    parser.add_argument('--data_dir', type=str, default='tic_tac_toe_advanced_data/npz_data',
                        help='包含 NPZ 数据文件的目录')
    parser.add_argument('--output_dir', type=str, default='tic_tac_toe_models',
                        help='保存模型输出的目录')
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'transformer', 'ensemble'],
                        help='要训练的模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='训练批量大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='初始学习率')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout 率')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--data_file', type=str, default=None,
                        help='要使用的特定 NPZ 文件（否则使用最新的）')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置 TensorFlow 内存增长以避免 OOM 错误
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"💻 找到 {len(physical_devices)} 个 GPU。已启用内存增长。")
        else:
            print("⚠️ 未找到 GPU。在 CPU 上训练。")
    except:
        print("⚠️ 设置 GPU 内存增长失败。")

    # 查找并加载数据
    try:
        if args.data_file:
            npz_file = args.data_file
        else:
            npz_file = find_latest_npz(args.data_dir)

        print(f"📂 从以下位置加载数据: {npz_file}")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(npz_file)

        print(f"训练集: {X_train.shape[0]} 个样本")
        print(f"验证集: {X_val.shape[0]} 个样本")
        print(f"测试集: {X_test.shape[0]} 个样本")
        print(f"输入形状: {X_train.shape[1:]}")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        exit(1)

    # 使用时间戳创建模型名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{timestamp}"

    # 构建模型
    print(f"🏗️ 构建 {args.model_type.upper()} 模型...")
    try:
        if args.model_type == 'cnn':
            model = build_cnn_model(input_shape=X_train.shape[1:], dropout_rate=args.dropout)
        elif args.model_type == 'transformer':
            model = build_transformer_model(
                input_shape=X_train.shape[1:],
                dropout_rate=args.dropout
            )
        elif args.model_type == 'ensemble':
            model = build_ensemble_model(
                input_shape=X_train.shape[1:],
                dropout_rate=args.dropout
            )
    except Exception as e:
        print(f"构建模型时出错: {str(e)}")
        print("将使用 CNN 模型作为备用...")
        model = build_cnn_model(input_shape=X_train.shape[1:], dropout_rate=args.dropout)

    # 编译模型
    model = compile_model(model, learning_rate=args.learning_rate, mixed_precision=args.mixed_precision)

    # 保存模型摘要
    save_model_summary(model, args.output_dir, model_name)

    # 创建回调
    callbacks = create_callbacks(args.output_dir, model_name, patience=args.patience)

    # 训练模型
    print(f"🚀 训练 {args.model_type.upper()} 模型，轮次数: {args.epochs}...")
    start_time = time.time()

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"⏱️ 训练完成，用时 {training_time:.2f} 秒")

        # 绘制训练历史
        plot_training_history(history, args.output_dir, model_name)

        # 评估模型
        results = evaluate_model(model, X_test, y_test, args.output_dir, model_name)

        # 打印最终结果
        print("\n📈 最终结果:")
        print(f"测试准确率: {results['metrics']['accuracy']:.4f}")
        print(f"测试损失: {results['metrics']['loss']:.4f}")

        # 保存模型元数据
        metadata = {
            'model_type': args.model_type,
            'input_shape': list(X_train.shape[1:]),
            'data_file': npz_file,
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'test_samples': X_test.shape[0],
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'training_time': training_time,
            'test_results': results['metrics'],
            'timestamp': timestamp,
            'tensorflow_version': tf.__version__
        }

        with open(os.path.join(args.output_dir, model_name, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # 尝试保存模型
        try:
            model_path = os.path.join(args.output_dir, model_name, 'final_model.h5')
            model.save(model_path)
            print(f"模型已保存到 {model_path}")
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            try:
                # 尝试使用另一种格式保存
                model_path = os.path.join(args.output_dir, model_name, 'final_model')
                model.save(model_path)
                print(f"模型已保存到 {model_path}")
            except:
                print("无法保存模型。")

        print(f"✅ 模型和输出已保存到 {os.path.join(args.output_dir, model_name)}")

    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        print("请检查您的数据和模型配置。")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"执行时出错: {str(e)}")
        import traceback
        traceback.print_exc()
