import numpy as np
import argparse
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from glob import glob

def find_latest_npz(data_dir):
    npz_files = glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in directory {data_dir}")
    return max(npz_files, key=os.path.getmtime)

def load_data(npz_file):
    data = np.load(npz_file)
    X_train = data["X_train"].reshape(-1, 3, 3, 2)
    X_test = data["X_test"].reshape(-1, 3, 3, 2)
    return X_train, data["y_train"], X_test, data["y_test"]

def build_cnn_model(input_shape=(3,3,2)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, kernel_size=(2,2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="tic_tac_toe_data/npz_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="tic_tac_toe_cnn_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    latest_npz_file = find_latest_npz(args.data_dir)
    print(f"✅ Using NPZ file: {latest_npz_file}")

    X_train, y_train, X_test, y_test = load_data(latest_npz_file)

    model = build_cnn_model()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'best_cnn_model.keras'),
                                 save_best_only=True, monitor='val_loss')

    history = model.fit(X_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, checkpoint])

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    plot_training(history, args.output_dir)

if __name__ == "__main__":
    main()
