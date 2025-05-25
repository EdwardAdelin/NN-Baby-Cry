"""
Generate synthetic data for initial testing of the baby cry detection model.
This script creates simple synthetic audio samples of baby cries and non-cries.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt

def generate_sine_wave(freq, duration, sample_rate=16000, amplitude=0.5):
    """
    Generate a sine wave.
    
    Parameters:
    -----------
    freq : float
        Frequency of the sine wave in Hz
    duration : float
        Duration of the sine wave in seconds
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
    amplitude : float, optional
        Amplitude of the sine wave, default is 0.5
        
    Returns:
    --------
    wave : numpy.ndarray
        Sine wave
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    return wave

def generate_white_noise(duration, sample_rate=16000, amplitude=0.1):
    """
    Generate white noise.
    
    Parameters:
    -----------
    duration : float
        Duration of the noise in seconds
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
    amplitude : float, optional
        Amplitude of the noise, default is 0.1
        
    Returns:
    --------
    noise : numpy.ndarray
        White noise
    """
    noise = amplitude * np.random.normal(0, 1, int(sample_rate * duration))
    return noise

def generate_synthetic_cry(duration, sample_rate=16000):
    """
    Generate a synthetic baby cry.
    A baby cry typically has a fundamental frequency around 400-600 Hz
    with harmonics and amplitude modulation.
    
    Parameters:
    -----------
    duration : float
        Duration of the cry in seconds
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
        
    Returns:
    --------
    cry : numpy.ndarray
        Synthetic baby cry
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Fundamental frequency that varies over time (400-600 Hz)
    f0 = 400 + 200 * np.sin(2 * np.pi * 0.5 * t)
    
    # Generate base cry with fundamental frequency
    cry = 0.5 * np.sin(2 * np.pi * f0 * t)
    
    # Add harmonics
    cry += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)  # First harmonic
    cry += 0.15 * np.sin(2 * np.pi * 3 * f0 * t)  # Second harmonic
    
    # Add amplitude modulation (tremolo)
    tremolo = 0.7 + 0.3 * np.sin(2 * np.pi * 8 * t)
    cry *= tremolo
    
    # Add some noise
    cry += generate_white_noise(duration, sample_rate, amplitude=0.05)
    
    # Normalize
    cry = 0.95 * cry / np.max(np.abs(cry))
    
    return cry

def generate_synthetic_non_cry(duration, sample_rate=16000):
    """
    Generate a synthetic non-cry sound.
    
    Parameters:
    -----------
    duration : float
        Duration of the sound in seconds
    sample_rate : int, optional
        Sample rate, default is 16000 Hz
        
    Returns:
    --------
    non_cry : numpy.ndarray
        Synthetic non-cry sound
    """
    # Generate a mixture of ambient sounds
    non_cry = generate_white_noise(duration, sample_rate, amplitude=0.2)
    
    # Add some low-frequency rumble
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    non_cry += 0.3 * np.sin(2 * np.pi * 100 * t)
    
    # Add random tones
    for _ in range(3):
        freq = np.random.uniform(200, 2000)
        start = int(np.random.uniform(0, 0.8) * sample_rate * duration)
        end = start + int(np.random.uniform(0.1, 0.5) * sample_rate * duration)
        if end > len(non_cry):
            end = len(non_cry)
        
        tone = 0.2 * np.sin(2 * np.pi * freq * t[0:end-start])
        non_cry[start:end] += tone
    
    # Normalize
    non_cry = 0.95 * non_cry / np.max(np.abs(non_cry))
    
    return non_cry

def plot_waveform(audio, sample_rate, title, output_path):
    """
    Plot a waveform.
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio time series
    sample_rate : int
        Sample rate
    title : str
        Plot title
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 4))
    time = np.arange(0, len(audio)) / sample_rate
    plt.plot(time, audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_dataset(args):
    """
    Generate a synthetic dataset of baby cries and non-cries.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create metadata DataFrame
    metadata = {
        'filename': [],
        'is_cry': []
    }
    
    print(f"Generating {args.num_samples} synthetic audio samples...")
    
    # Generate samples
    for i in range(args.num_samples):
        # Determine if this sample is a cry
        is_cry = i < args.num_samples // 2
        
        # Generate audio
        if is_cry:
            audio = generate_synthetic_cry(args.duration, args.sample_rate)
            filename = f"cry_{i:04d}.wav"
        else:
            audio = generate_synthetic_non_cry(args.duration, args.sample_rate)
            filename = f"non_cry_{i:04d}.wav"
        
        # Convert to int16
        audio_int16 = np.int16(audio * 32767)
        
        # Save audio file
        audio_path = os.path.join(audio_dir, filename)
        wavfile.write(audio_path, args.sample_rate, audio_int16)
        
        # Plot waveform for first few samples
        if i < 5:
            plot_path = os.path.join(plots_dir, f"{os.path.splitext(filename)[0]}_waveform.png")
            title = "Baby Cry" if is_cry else "Non-Cry"
            plot_waveform(audio, args.sample_rate, title, plot_path)
        
        # Add to metadata
        metadata['filename'].append(filename)
        metadata['is_cry'].append(1 if is_cry else 0)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_samples} samples")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Generated {args.num_samples} synthetic audio samples")
    print(f"Metadata saved to {metadata_path}")
    print(f"Audio files saved to {audio_dir}")
    print(f"Plots saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for baby cry detection')
    parser.add_argument('--output_dir', type=str, default='../data', help='Directory to save the data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate (half cry, half non-cry)')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration of each sample in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate in Hz')
    
    args = parser.parse_args()
    
    generate_dataset(args)

if __name__ == "__main__":
    main()
