"""
Training script for the baby cry detection CNN model.
This script handles the training process and saves the trained model.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from data_preprocessing import prepare_dataset, save_preprocessed_data, load_preprocessed_data
from model import create_simple_cnn_model, create_deeper_cnn_model, get_callbacks

def plot_training_history(history, output_dir):
    """
    Plot the training and validation metrics.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history
    output_dir : str
        Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Plot precision and recall if available
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'))
        plt.close()

def train_model(args):
    """
    Train the CNN model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"baby_cry_cnn_{args.model_type}_{timestamp}.h5"
    model_save_path = os.path.join(args.model_dir, model_name)
    
    # Check if preprocessed data exists
    preprocessed_path = os.path.join(args.output_dir, 'preprocessed')
    if os.path.exists(preprocessed_path) and not args.force_preprocess:
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = load_preprocessed_data(preprocessed_path)
    else:
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = prepare_dataset(
            args.data_dir, 
            args.metadata_file,
            sample_rate=args.sample_rate,
            duration=args.duration,
            n_mels=args.n_mels
        )
        # Save preprocessed data
        save_preprocessed_data(X_train, X_test, y_train, y_test, preprocessed_path)
    
    # Get model input shape
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # Create model
    if args.model_type == 'simple':
        model = create_simple_cnn_model(input_shape)
    else:
        model = create_deeper_cnn_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks(model_save_path)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, args.output_dir)
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    
    # Save evaluation metrics
    with open(os.path.join(args.output_dir, f"evaluation_{timestamp}.txt"), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test loss: {loss:.4f}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Test precision: {precision:.4f}\n")
        f.write(f"Test recall: {recall:.4f}\n")
    
    print(f"Model saved to: {model_save_path}")
    print(f"Training history and evaluation saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model for baby cry detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--metadata_file', type=str, required=True, help='CSV file with filenames and labels')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save the model')
    parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save output files')
    parser.add_argument('--model_type', type=str, choices=['simple', 'deeper'], default='simple', help='Type of CNN model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds to extract from each audio file')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    parser.add_argument('--force_preprocess', action='store_true', help='Force reprocessing of data even if preprocessed data exists')
    
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main()
