from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import os
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_latest_model(model_dir, pattern="*.h5"):
    """自动寻找指定目录中最新的模型文件"""
    model_files = glob.glob(os.path.join(model_dir, pattern))
    if not model_files:
        raise FileNotFoundError(f"No model files found in directory: {model_dir}")
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

# 加载最新的训练好的模型
MODEL_DIR = "./gomoku_data/models"  # 指定模型目录
latest_model_path = find_latest_model(MODEL_DIR)
model = load_model(latest_model_path, compile=False)

print(f"Loaded model: {latest_model_path}")

class BoardState(BaseModel):
    board: list  # 二维数组表示棋盘状态，0表示空位，1表示黑子，2表示白子
    current_player: int  # 当前玩家，1或2

@app.post("/predict")
def predict_move(state: BoardState):
    board_array = np.array(state.board, dtype=np.float32)

    # 调整输入数据以匹配模型训练时的三通道输入
    input_data = np.zeros((1, board_array.shape[0], board_array.shape[1], 3), dtype=np.float32)

    # 设置黑子通道
    input_data[0, :, :, 0] = (board_array == 1)
    # 设置白子通道
    input_data[0, :, :, 1] = (board_array == 2)
    # 设置空位通道
    input_data[0, :, :, 2] = (board_array == 0)

    predictions = model.predict(input_data)

    move_probs = predictions[0].reshape(-1)
    sorted_indices = np.argsort(-move_probs)

    for idx in sorted_indices:
        row, col = divmod(idx, board_array.shape[1])
        if board_array[row, col] == 0:
            return {"row": int(row), "col": int(col)}

    return {"error": "No valid moves available"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
