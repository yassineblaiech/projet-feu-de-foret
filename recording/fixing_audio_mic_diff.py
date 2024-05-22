import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# Function to normalize audio
def normalize_audio(audio, target_dBFS=-20):
    rms = np.sqrt(np.mean(audio**2))
    multiplier = 10 ** (target_dBFS / 20) / rms
    normalized_audio = audio * multiplier
    return normalized_audio

# Function to apply a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Load, preprocess, and save audio
def preprocess_and_save_audio(file_path, output_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=None)

    # Normalize audio
    audio = normalize_audio(audio)

    # Apply bandpass filter
    audio = bandpass_filter(audio, 400, 3000, sr)

    # Save audio
    sf.write(output_path, audio, sr)
    print(f"Audio saved to {output_path}")

# Example usage
input_audio_path = 'test.wav'
output_audio_path = 'test_changed.wav'
preprocess_and_save_audio(input_audio_path, output_audio_path)