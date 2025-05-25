"""
Main entry point for the baby cry detection system.
This script serves as a central hub for all functionality in the project.
"""

import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Baby Cry Detection System using CNN',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate synthetic data command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic data for testing')
    generate_parser.add_argument('--output_dir', type=str, default='../data', help='Directory to save the data')
    generate_parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate (half cry, half non-cry)')
    generate_parser.add_argument('--duration', type=float, default=5.0, help='Duration of each sample in seconds')
    generate_parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate in Hz')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the CNN model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Directory containing audio files')
    train_parser.add_argument('--metadata_file', type=str, required=True, help='CSV file with filenames and labels')
    train_parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save the model')
    train_parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save output files')
    train_parser.add_argument('--model_type', type=str, choices=['simple', 'deeper'], default='simple', help='Type of CNN model to train')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    train_parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds to extract from each audio file')
    train_parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    train_parser.add_argument('--force_preprocess', action='store_true', help='Force reprocessing of data even if preprocessed data exists')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    evaluate_parser.add_argument('--preprocessed_data', type=str, required=True, help='Directory containing preprocessed data')
    evaluate_parser.add_argument('--output_dir', type=str, default='../output/evaluation', help='Directory to save evaluation results')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on audio files')
    inference_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    inference_parser.add_argument('--input_path', type=str, required=True, help='Path to an audio file or directory containing audio files')
    inference_parser.add_argument('--output_file', type=str, help='Path to save the results')
    inference_parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    inference_parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds to extract from each audio file')
    inference_parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    
    # Realtime detection command
    realtime_parser = subparsers.add_parser('realtime', help='Run real-time baby cry detection')
    realtime_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    realtime_parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    realtime_parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    realtime_parser.add_argument('--window_duration', type=float, default=5.0, help='Duration of the analysis window in seconds')
    realtime_parser.add_argument('--overlap', type=float, default=2.0, help='Overlap between consecutive windows in seconds')
    realtime_parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Add the src directory to the Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Execute the selected command
    if args.command == 'generate':
        from generate_synthetic_data import generate_dataset
        generate_dataset(args)
    elif args.command == 'train':
        from train import train_model
        train_model(args)
    elif args.command == 'evaluate':
        from evaluate import evaluate_model
        evaluate_model(args)
    elif args.command == 'inference':
        from inference import run_inference
        run_inference(args)
    elif args.command == 'realtime':
        from realtime_detection import RealTimeDetector
        try:
            print("Starting real-time baby cry detection...")
            print("Press Ctrl+C to stop")
            detector = RealTimeDetector(
                args.model_path,
                sample_rate=args.sample_rate,
                n_mels=args.n_mels,
                window_duration=args.window_duration,
                overlap=args.overlap,
                threshold=args.threshold
            )
            detector.start()
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            if 'detector' in locals():
                detector.stop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
