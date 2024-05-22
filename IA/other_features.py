import librosa
import numpy as np
from log_mel_spectro import audio_to_spectrogram
def extract_audio_features(file_path, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectrogram=audio_to_spectrogram(file_path)
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Extract 13 MFCCs
    
    # Rhythmic Features - Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Time-Domain Features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    rms_energy = librosa.feature.rms(y=y)
    
    # Compile all the extracted features into a dictionary
    features = {
    'spectrogram': spectrogram,
    'spectral_centroid': spectral_centroid,
    'spectral_bandwidth': spectral_bandwidth,
    'spectral_rolloff': spectral_rolloff,
    'zcr': zero_crossing_rate,
    'rms_energy': rms_energy,
    'spectral_contrast_input': spectral_contrast,
    'mfccs_input': mfccs,
    'tempo_input': tempo
    }
    
    return features

def feature_names():
    return ['spectrogram', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy', 'spectral_contrast_input', 'mfccs_input', 'tempo_input']

if __name__=='__main__':
    # Example usage
    file_path = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/fire_test/videoplayback (1)_01.wav'
    features = extract_audio_features(file_path)
    print(features)
    for key in features.keys():
        print(key,' has the shape: ',features[key].shape)