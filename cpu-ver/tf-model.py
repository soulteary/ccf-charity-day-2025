import os
import tensorflow as tf
import subprocess


def find_latest_model(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        raise FileNotFoundError("No .h5 model files found in the specified directory.")
    model_files.sort(reverse=True)
    return os.path.join(model_dir, model_files[0])

# 自动加载最新模型文件并导出 SavedModel 格式
latest_model_path = find_latest_model("gomoku_data/models")
model = tf.keras.models.load_model(latest_model_path)

saved_model_dir = "gomoku_data/saved_models/latest_model"
os.makedirs(saved_model_dir, exist_ok=True)
model.export(saved_model_dir)
print(f"模型已保存至: {saved_model_dir}")

# 转换为TensorFlow.js格式
tfjs_output_dir = "gomoku_data/tfjs_model/latest_model"
os.makedirs(tfjs_output_dir, exist_ok=True)
subprocess.run([
    "tensorflowjs_converter",
    "--input_format", "tf_saved_model",
    "--output_format", "tfjs_graph_model",
    saved_model_dir,
    tfjs_output_dir
], check=True)

print(f"TensorFlow.js 模型已保存至: {tfjs_output_dir}")
