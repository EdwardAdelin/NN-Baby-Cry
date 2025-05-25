"""
Evaluation script for the baby cry detection CNN model.
This script evaluates a trained model on the test dataset and generates performance metrics.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score
)
import seaborn as sns

# Import local modules
from data_preprocessing import load_preprocessed_data

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Plot the confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    output_dir : str
        Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, output_dir):
    """
    Plot the precision-recall curve.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    output_dir : str
        Directory to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """
    Plot the ROC curve.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred_proba : numpy.ndarray
        Predicted probabilities
    output_dir : str
        Directory to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Random prediction line
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def evaluate_model(args):
    """
    Evaluate the CNN model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Print model summary
    model.summary()
    
    # Check if preprocessed data exists
    preprocessed_path = args.preprocessed_data
    if not os.path.exists(preprocessed_path):
        print(f"Preprocessed data not found at {preprocessed_path}")
        print("Please run the training script first or provide the correct path.")
        return
    
    # Load test data
    print(f"Loading test data from {preprocessed_path}...")
    _, X_test, _, y_test = load_preprocessed_data(preprocessed_path)
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test = y_test.flatten()
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, args.output_dir)
    
    # Plot precision-recall curve
    print("Plotting precision-recall curve...")
    plot_precision_recall_curve(y_test, y_pred_proba, args.output_dir)
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(y_test, y_pred_proba, args.output_dir)
    
    # Save evaluation metrics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"detailed_evaluation_{timestamp}.txt"), 'w') as f:
        f.write("===== Evaluation Results =====\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Evaluation results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN model for baby cry detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--preprocessed_data', type=str, required=True, help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='../output/evaluation', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == "__main__":
    main()
