"""
Data preprocessing module for the baby cry detection CNN.
This module contains functions for loading and preprocessing audio files for training and testing.
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_audio_file(file_path, sample_rate=16000, duration=None):
    """
    Load an audio file and resample it to the target sample rate.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    sample_rate : int, optional
        Target sample rate, default is 16000 Hz
    duration : float or None, optional
        Duration in seconds to load, or None to load the entire file
        
    Returns:
    --------
    audio : numpy.ndarray
        Audio time series
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_mel_spectrogram(audio, sample_rate=16000, n_mels=128, hop_length=512):
    """
    Extract mel spectrogram features from an audio signal.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio time series
    sample_rate : int, optional
        Sample rate of the audio signal, default is 16000 Hz
    n_mels : int, optional
        Number of mel bands, default is 128
    hop_length : int, optional
        Number of samples between frames, default is 512
        
    Returns:
    --------
    log_mel_spec : numpy.ndarray
        Log-scaled mel spectrogram
    """
    if audio is None or len(audio) == 0:
        return None
    
    # Create mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_mels=n_mels,
        hop_length=hop_length
    )
    
    # Convert to logarithmic scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def pad_or_trim_audio(audio, target_length):
    """
    Pad or trim audio to a target length.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio time series
    target_length : int
        Target length in samples
        
    Returns:
    --------
    processed_audio : numpy.ndarray
        Audio time series with length equal to target_length
    """
    if len(audio) > target_length:
        # Trim audio
        return audio[:target_length]
    elif len(audio) < target_length:
        # Pad audio with zeros
        return np.pad(audio, (0, target_length - len(audio)))
    else:
        return audio

def visualize_spectrogram(spectrogram, title='Mel Spectrogram'):
    """
    Visualize a spectrogram.
    
    Parameters:
    -----------
    spectrogram : numpy.ndarray
        Spectrogram to visualize
    title : str, optional
        Plot title, default is 'Mel Spectrogram'
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram, 
        x_axis='time', 
        y_axis='mel', 
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def prepare_dataset(data_dir, csv_path, sample_rate=16000, duration=5, n_mels=128):
    """
    Process audio files and create a dataset for training/testing.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing audio files
    csv_path : str
        Path to CSV file with filenames and labels
    sample_rate : int, optional
        Target sample rate, default is 16000 Hz
    duration : float, optional
        Duration in seconds to load for each file, default is 5
    n_mels : int, optional
        Number of mel bands, default is 128
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Training and testing data and labels
    """
    print(f"Preparing dataset from {data_dir} using metadata from {csv_path}")
    
    features = []
    labels = []
    
    # Load metadata
    df = pd.read_csv(csv_path)
    total_files = len(df)
    
    print(f"Processing {total_files} files...")
    
    for idx, row in df.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        
        if os.path.exists(file_path):
            # Print progress
            if idx % 10 == 0:
                print(f"Processing file {idx}/{total_files}: {row['filename']}")
                
            # Load audio with fixed duration
            target_samples = int(duration * sample_rate)
            audio = load_audio_file(file_path, sample_rate=sample_rate)
            
            if audio is not None:
                # Ensure consistent length
                audio = pad_or_trim_audio(audio, target_samples)
                
                # Extract mel spectrogram
                mel_spec = extract_mel_spectrogram(audio, sample_rate=sample_rate, n_mels=n_mels)
                
                if mel_spec is not None:
                    features.append(mel_spec)
                    labels.append(1 if row['is_cry'] else 0)  # Binary label: 1 for cry, 0 for not cry
        else:
            print(f"File not found: {file_path}")
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Dataset shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Reshape for CNN input (samples, height, width, channels)
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Save preprocessed data to files.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Training and testing data and labels
    output_dir : str
        Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"Preprocessed data saved to {output_dir}")

def load_preprocessed_data(input_dir):
    """
    Load preprocessed data from files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the preprocessed data files
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Training and testing data and labels
    """
    X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    
    print(f"Loaded preprocessed data from {input_dir}")
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for preprocessing audio data for baby cry detection.")
    print("Import and use these functions in your training script.")
