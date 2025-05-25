"""
Inference script for the baby cry detection CNN model.
This script loads a trained model and uses it to predict whether an audio file contains a baby cry.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from datetime import datetime

# Import local modules
from data_preprocessing import (
    load_audio_file, extract_mel_spectrogram,
    pad_or_trim_audio, visualize_spectrogram
)

def predict_single_file(file_path, model, sample_rate=16000, duration=5, n_mels=128, plot=False):
    """
    Make a prediction for a single audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    model : tensorflow.keras.models.Model
        Trained model
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
    duration : float, optional
        Duration in seconds to extract from the audio file, default is 5
    n_mels : int, optional
        Number of mel bands, default is 128
    plot : bool, optional
        Whether to plot the mel spectrogram, default is False
        
    Returns:
    --------
    is_cry : bool
        Whether the audio file contains a baby cry
    confidence : float
        Confidence score (probability)
    """
    print(f"Processing file: {file_path}")
    
    # Load audio
    audio = load_audio_file(file_path, sample_rate=sample_rate, duration=duration)
    if audio is None:
        print(f"Error: Could not load audio file {file_path}")
        return None, 0.0
    
    # Pad or trim audio to fixed length
    target_samples = int(duration * sample_rate)
    audio = pad_or_trim_audio(audio, target_samples)
    
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sample_rate=sample_rate, n_mels=n_mels)
    if mel_spec is None:
        print(f"Error: Could not extract mel spectrogram from {file_path}")
        return None, 0.0
    
    # Reshape for model input (adding batch and channel dimensions)
    X = np.expand_dims(np.expand_dims(mel_spec, axis=0), axis=-1)
    
    # Plot spectrogram if requested
    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_spec,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.show()
    
    # Make prediction
    pred_prob = model.predict(X, verbose=0)[0][0]
    is_cry = bool(pred_prob >= 0.5)
    
    return is_cry, pred_prob

def predict_batch(file_list, model, sample_rate=16000, duration=5, n_mels=128):
    """
    Make predictions for a batch of audio files.
    
    Parameters:
    -----------
    file_list : list
        List of paths to audio files
    model : tensorflow.keras.models.Model
        Trained model
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
    duration : float, optional
        Duration in seconds to extract from each audio file, default is 5
    n_mels : int, optional
        Number of mel bands, default is 128
        
    Returns:
    --------
    results : list
        List of dictionaries with file paths, predictions, and confidence scores
    """
    results = []
    
    for file_path in file_list:
        is_cry, confidence = predict_single_file(
            file_path, model, 
            sample_rate=sample_rate, 
            duration=duration, 
            n_mels=n_mels
        )
        
        if is_cry is not None:
            result = {
                'file_path': file_path,
                'is_cry': is_cry,
                'confidence': confidence
            }
            results.append(result)
            
            # Print prediction
            status = "BABY CRY" if is_cry else "NO CRY"
            print(f"Prediction: {status} (confidence: {confidence:.4f})")
    
    return results

def save_results(results, output_file):
    """
    Save prediction results to a file.
    
    Parameters:
    -----------
    results : list
        List of dictionaries with file paths, predictions, and confidence scores
    output_file : str
        Path to the output file
    """
    with open(output_file, 'w') as f:
        f.write("file_path,is_cry,confidence\n")
        for result in results:
            f.write(f"{result['file_path']},{result['is_cry']},{result['confidence']:.4f}\n")
    
    print(f"Results saved to {output_file}")

def run_inference(args):
    """
    Run inference on audio files.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Create output directory if it doesn't exist
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Get list of audio files
    if os.path.isdir(args.input_path):
        print(f"Scanning directory {args.input_path} for audio files...")
        file_list = []
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    file_list.append(os.path.join(root, file))
    else:
        file_list = [args.input_path]
    
    print(f"Found {len(file_list)} audio files")
    
    # Make predictions
    results = predict_batch(
        file_list, model,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels
    )
    
    # Save results if output file is specified
    if args.output_file:
        save_results(results, args.output_file)

def main():
    parser = argparse.ArgumentParser(description='Predict baby cries in audio files using a trained CNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--input_path', type=str, required=True, help='Path to an audio file or directory containing audio files')
    parser.add_argument('--output_file', type=str, help='Path to save the results')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds to extract from each audio file')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    
    args = parser.parse_args()
    
    run_inference(args)

if __name__ == "__main__":
    main()
