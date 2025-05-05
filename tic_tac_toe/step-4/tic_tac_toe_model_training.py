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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def find_latest_npz(data_dir):
    """Find the most recent NPZ file in the given directory"""
    npz_files = glob(os.path.join(data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in directory {data_dir}")
    return max(npz_files, key=os.path.getmtime)

def load_data(npz_file):
    """Load data from NPZ file with validation split support"""
    data = np.load(npz_file, allow_pickle=True)
    
    # Check if the file contains X_val and y_val
    if "X_val" in data and "y_val" in data:
        return (
            data["X_train"], data["y_train"], 
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"]
        )
    else:
        # Backward compatibility for older data files
        print("⚠️ No validation split found in data file. Using test set as validation.")
        return (
            data["X_train"], data["y_train"], 
            data["X_test"], data["y_test"],
            data["X_test"], data["y_test"]
        )

def build_cnn_model(input_shape=(3, 3, 6), dropout_rate=0.3):
    """Build a CNN model for Tic Tac Toe prediction"""
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Second convolutional block
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer (win, loss, draw)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_transformer_model(input_shape=(3, 3, 6), num_heads=4, ff_dim=128, dropout_rate=0.2):
    """Build a transformer-based model for Tic Tac Toe prediction"""
    inputs = Input(shape=input_shape)
    
    # Reshape 3x3 board into a sequence
    # First path: flatten and process with transformer
    x1 = Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)
    
    # Add positional encoding (simple learned embedding)
    pos_encoding = tf.range(start=0, limit=input_shape[0] * input_shape[1], delta=1)
    pos_encoding = tf.one_hot(pos_encoding, depth=ff_dim)
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    
    # Project input to higher dimension
    x1 = Dense(ff_dim)(x1)
    x1 = x1 + pos_encoding
    
    # Apply multi-head attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=ff_dim // num_heads
    )(x1, x1)
    attention_output = Dropout(dropout_rate)(attention_output)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + attention_output)
    
    # Feed forward network
    ffn_output = Dense(ff_dim * 2, activation='relu')(x1)
    ffn_output = Dense(ff_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    x1 = LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
    
    # Second path: use convolutional features
    x2 = Conv2D(64, (2, 2), activation='relu', padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(ff_dim, (2, 2), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)
    
    # Combine both paths
    combined = Concatenate()([Flatten()(x1), x2])
    combined = Dense(ff_dim, activation='relu')(combined)
    combined = LayerNormalization(epsilon=1e-6)(combined)
    combined = Dropout(dropout_rate)(combined)
    
    # Output layer
    outputs = Dense(3, activation='softmax')(combined)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model, learning_rate=3e-4, mixed_precision=False):
    """Compile the model with appropriate optimizer and metrics"""
    if mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision training")
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

def create_callbacks(output_dir, model_name, patience=15):
    """Create callbacks for training"""
    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.keras'),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 3,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=os.path.join(model_dir, 'logs'),
        histogram_freq=1,
        write_graph=True
    )
    
    csv_logger = CSVLogger(
        os.path.join(model_dir, 'training_log.csv'),
        separator=',',
        append=False
    )
    
    return [early_stop, checkpoint, reduce_lr, tensorboard, csv_logger]

def plot_training_history(history, output_dir, model_name):
    """Plot training metrics and save figures"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, model_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Accuracy and loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'accuracy_loss.png'), dpi=300)
    plt.close()
    
    # Additional metrics plot if available
    if 'auc' in history.history:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['auc'], label='Train')
        plt.plot(history.history['val_auc'], label='Validation')
        plt.title('AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['precision'], label='Train')
        plt.plot(history.history['val_precision'], label='Validation')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['recall'], label='Train')
        plt.plot(history.history['val_recall'], label='Validation')
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'additional_metrics.png'), dpi=300)
        plt.close()

def evaluate_model(model, X_test, y_test, output_dir, model_name):
    """Evaluate model performance and save results"""
    # Create evaluation directory
    eval_dir = os.path.join(output_dir, model_name, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate model
    print("\n📊 Evaluating model on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Win', 'Loss', 'Draw'],
                yticklabels=['Win', 'Loss', 'Draw'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Classification report
    class_report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=['Win', 'Loss', 'Draw'],
        output_dict=True
    )
    
    # Save results
    results = {
        'metrics': {name: float(value) for name, value in zip(model.metrics_names, test_results)},
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def save_model_summary(model, output_dir, model_name):
    """Save model architecture summary to a text file"""
    summary_path = os.path.join(output_dir, model_name, 'model_summary.txt')
    
    # Redirect stdout to capture summary
    import io
    import sys
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Print model summary
    model.summary()
    
    # Restore stdout and save captured output
    sys.stdout = original_stdout
    with open(summary_path, 'w') as f:
        f.write(captured_output.getvalue())

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Tic Tac Toe prediction model')
    parser.add_argument('--data_dir', type=str, default='tic_tac_toe_advanced_data/npz_data',
                        help='Directory containing NPZ data files')
    parser.add_argument('--output_dir', type=str, default='tic_tac_toe_models',
                        help='Directory to save model outputs')
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['cnn', 'transformer', 'ensemble'],
                        help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Specific NPZ file to use (otherwise latest is used)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorFlow memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"💻 Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    else:
        print("⚠️ No GPU found. Training on CPU.")
    
    # Find and load data
    if args.data_file:
        npz_file = args.data_file
    else:
        npz_file = find_latest_npz(args.data_dir)
    
    print(f"📂 Loading data from: {npz_file}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(npz_file)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input shape: {X_train.shape[1:]}")
    
    # Create model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{timestamp}"
    
    # Build model
    print(f"🏗️ Building {args.model_type.upper()} model...")
    if args.model_type == 'cnn':
        model = build_cnn_model(input_shape=X_train.shape[1:], dropout_rate=args.dropout)
    elif args.model_type == 'transformer':
        model = build_transformer_model(
            input_shape=X_train.shape[1:],
            dropout_rate=args.dropout
        )
    elif args.model_type == 'ensemble':
        # Create ensemble of CNN and Transformer
        cnn_model = build_cnn_model(input_shape=X_train.shape[1:], dropout_rate=args.dropout)
        transformer_model = build_transformer_model(
            input_shape=X_train.shape[1:],
            dropout_rate=args.dropout
        )
        
        # Create ensemble input
        inputs = Input(shape=X_train.shape[1:])
        
        # Get outputs from both models
        cnn_output = cnn_model(inputs)
        transformer_output = transformer_model(inputs)
        
        # Combine outputs
        combined = Concatenate()([cnn_output, transformer_output])
        combined = Dense(32, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(args.dropout)(combined)
        outputs = Dense(3, activation='softmax')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model = compile_model(model, learning_rate=args.learning_rate, mixed_precision=args.mixed_precision)
    
    # Save model summary
    save_model_summary(model, args.output_dir, model_name)
    
    # Create callbacks
    callbacks = create_callbacks(args.output_dir, model_name, patience=args.patience)
    
    # Train model
    print(f"🚀 Training {args.model_type.upper()} model for {args.epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    plot_training_history(history, args.output_dir, model_name)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, args.output_dir, model_name)
    
    # Print final results
    print("\n📈 Final Results:")
    print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Test Loss: {results['metrics']['loss']:.4f}")
    
    # Save model metadata
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
    
    print(f"✅ Model and outputs saved to {os.path.join(args.output_dir, model_name)}")

if __name__ == '__main__':
    main()
