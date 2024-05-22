import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def audio_to_spectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Normalize
    S_dB = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    spectrogram=np.array([[round(num,2) for num in row] for row in S_dB])
    return spectrogram


if __name__=='__main__':
    # Example usage
    file_path = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/fire_test/videoplayback (1)_01.wav'
    log_mel_spectrogram = audio_to_spectrogram(file_path)
    print(log_mel_spectrogram.shape)