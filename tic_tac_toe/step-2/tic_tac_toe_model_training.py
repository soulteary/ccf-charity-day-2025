import numpy as np
import argparse
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from glob import glob

def find_latest_npz(data_dir):
    npz_files = glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in directory {data_dir}")
    latest_file = max(npz_files, key=os.path.getmtime)
    return latest_file

def load_data(npz_file):
    data = np.load(npz_file)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training(history, output_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.title('Training History')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Automatically find latest NPZ data and train Tic-Tac-Toe model")
    parser.add_argument("--data_dir", type=str, default="tic_tac_toe_training_data/npz_data",
                        help="Directory containing NPZ data files")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="tic_tac_toe_model_output",
                        help="Directory to save trained model and results")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    latest_npz_file = find_latest_npz(args.data_dir)
    print(f"✅ Using NPZ file: {latest_npz_file}")

    X_train, y_train, X_test, y_test = load_data(latest_npz_file)

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_model.keras'),
        save_best_only=True,
        monitor='val_loss'
    )

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr, model_checkpoint]
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")

    plot_training(history, args.output_dir)

if __name__ == "__main__":
    main()