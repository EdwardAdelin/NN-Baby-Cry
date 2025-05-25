"""
Real-time baby cry detection using a trained CNN model.
This script uses the microphone to detect baby cries in real-time.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import librosa

# Import local modules
from data_preprocessing import extract_mel_spectrogram

class RealTimeDetector:
    """
    Real-time baby cry detection using a trained CNN model.
    """
    def __init__(self, model_path, sample_rate=16000, n_mels=128, 
                 window_duration=5, overlap=2, threshold=0.5):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file
        sample_rate : int, optional
            Sample rate, default is 16000 Hz
        n_mels : int, optional
            Number of mel bands, default is 128
        window_duration : float, optional
            Duration of the analysis window in seconds, default is 5
        overlap : float, optional
            Overlap between consecutive windows in seconds, default is 2
        threshold : float, optional
            Threshold for binary classification, default is 0.5
        """
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_duration = window_duration
        self.overlap = overlap
        self.threshold = threshold
        self.window_samples = int(window_duration * sample_rate)
        
        # Initialize buffers
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.audio_queue = queue.Queue()
        self.prediction_history = []
        self.max_history = 100
        
        # Animation parameters
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.mel_img = None
        self.prob_line = None
        self.threshold_line = None
        self.time_data = np.zeros(self.max_history)
        self.prob_data = np.zeros(self.max_history)
        
        # Thread controls
        self.running = False
        self.audio_thread = None
        self.prediction_thread = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio stream.
        
        Parameters:
        -----------
        indata : numpy.ndarray
            Input audio data
        frames : int
            Number of frames
        time_info : dict
            Time information
        status : sd.CallbackFlags
            Status flags
        """
        if status:
            print(f"Status: {status}")
        
        # Put the audio data in the queue
        self.audio_queue.put(indata.copy())
    
    def update_buffer(self, new_data):
        """
        Update the audio buffer with new data.
        
        Parameters:
        -----------
        new_data : numpy.ndarray
            New audio data
        """
        # Shift the buffer
        shift = len(new_data)
        self.audio_buffer = np.roll(self.audio_buffer, -shift)
        
        # Add new data
        self.audio_buffer[-shift:] = new_data.flatten()
    
    def predict_from_buffer(self):
        """
        Make a prediction from the current audio buffer.
        
        Returns:
        --------
        is_cry : bool
            Whether a baby cry is detected
        confidence : float
            Confidence score (probability)
        """
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(
            self.audio_buffer, 
            sample_rate=self.sample_rate, 
            n_mels=self.n_mels
        )
        
        if mel_spec is None:
            return False, 0.0
        
        # Reshape for model input (adding batch and channel dimensions)
        X = np.expand_dims(np.expand_dims(mel_spec, axis=0), axis=-1)
        
        # Make prediction
        pred_prob = self.model.predict(X, verbose=0)[0][0]
        is_cry = bool(pred_prob >= self.threshold)
        
        # Store spectrogram for visualization
        self.current_spectrogram = mel_spec
        
        return is_cry, pred_prob
    
    def prediction_loop(self):
        """
        Main prediction loop.
        """
        while self.running:
            try:
                # Get new audio data from the queue
                new_data = self.audio_queue.get(timeout=1)
                
                # Update the buffer
                self.update_buffer(new_data)
                
                # Make prediction
                is_cry, confidence = self.predict_from_buffer()
                
                # Add to prediction history
                timestamp = time.time()
                self.prediction_history.append((timestamp, confidence, is_cry))
                
                # Keep only the most recent predictions
                if len(self.prediction_history) > self.max_history:
                    self.prediction_history.pop(0)
                
                # Print prediction
                status = "BABY CRY DETECTED" if is_cry else "No cry"
                print(f"{status} (confidence: {confidence:.4f})")
                
            except queue.Empty:
                pass  # No new audio data
    
    def init_plot(self):
        """
        Initialize the visualization plot.
        """
        # Initialize spectrogram plot
        self.mel_img = self.ax1.imshow(
            np.zeros((self.n_mels, 100)),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        self.ax1.set_title('Mel Spectrogram')
        self.ax1.set_ylabel('Mel Bands')
        self.ax1.set_xlabel('Time')
        
        # Initialize prediction probability plot
        self.prob_line, = self.ax2.plot(self.time_data, self.prob_data, 'b-')
        self.threshold_line, = self.ax2.plot(
            [self.time_data[0], self.time_data[-1]], 
            [self.threshold, self.threshold], 
            'r--'
        )
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Prediction Probability')
        self.ax2.set_ylabel('Probability')
        self.ax2.set_xlabel('Time')
        
        return self.mel_img, self.prob_line, self.threshold_line
    
    def update_plot(self, frame):
        """
        Update the visualization plot.
        
        Parameters:
        -----------
        frame : int
            Animation frame
        
        Returns:
        --------
        artists : tuple
            Updated plot artists
        """
        # Update spectrogram
        if hasattr(self, 'current_spectrogram'):
            self.mel_img.set_array(self.current_spectrogram)
            self.mel_img.set_clim(vmin=np.min(self.current_spectrogram), 
                                 vmax=np.max(self.current_spectrogram))
        
        # Update prediction probability
        if self.prediction_history:
            recent = self.prediction_history[-min(self.max_history, len(self.prediction_history)):]
            times = [t for t, _, _ in recent]
            probs = [p for _, p, _ in recent]
            
            # Normalize times for display
            if len(times) > 1:
                norm_times = [(t - times[0]) for t in times]
            else:
                norm_times = [0]
            
            # Update data
            self.time_data = np.array(norm_times)
            self.prob_data = np.array(probs)
            
            # Update line
            self.prob_line.set_data(self.time_data, self.prob_data)
            
            # Update threshold line
            self.threshold_line.set_data([self.time_data[0], self.time_data[-1]], 
                                       [self.threshold, self.threshold])
            
            # Adjust x limits
            self.ax2.set_xlim(self.time_data[0], max(self.time_data[-1], 1))
        
        return self.mel_img, self.prob_line, self.threshold_line
    
    def start(self):
        """
        Start real-time detection.
        """
        # Set running flag
        self.running = True
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self.prediction_loop)
        self.prediction_thread.start()
        
        # Initialize the audio stream
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * (self.window_duration - self.overlap) / 2),
            dtype=np.float32
        )
        self.audio_stream.start()
        
        # Start animation
        self.init_plot()
        self.anim = FuncAnimation(
            self.fig, self.update_plot, interval=200,
            blit=True
        )
        plt.tight_layout()
        plt.show()
    
    def stop(self):
        """
        Stop real-time detection.
        """
        self.running = False
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Wait for threads to finish
        if self.prediction_thread:
            self.prediction_thread.join()

def main():
    parser = argparse.ArgumentParser(description='Real-time baby cry detection using a trained CNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel frequency bands')
    parser.add_argument('--window_duration', type=float, default=5.0, help='Duration of the analysis window in seconds')
    parser.add_argument('--overlap', type=float, default=2.0, help='Overlap between consecutive windows in seconds')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()
