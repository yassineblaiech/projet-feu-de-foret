import os
import librosa
import numpy as np
from scipy.signal import lfilter, firwin
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from scipy.signal import butter, lfilter

# Equalization - Focuses on emphasizing frequencies important for fire crackling sounds
def apply_equalization(y, sr):
    D = librosa.stft(y)
    hz_per_bin = sr / D.shape[0]
    # Assuming fire crackles are prominent around 110 and 31750 hz
    low_bin = int(110 / hz_per_bin)     #110 hz
    high_bin = int(31750/ hz_per_bin)   #31750 hz
    D[low_bin:high_bin, :] *= 20
    return librosa.istft(D)

def save_processed_audio(audio, output_file_path,sr):
    sf.write(output_file_path, audio,sr, format='WAV')

def process_audio_files(audio_repo, output_dir):
    for filename in os.listdir(audio_repo):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_repo, filename)
            audio, sr = librosa.load(file_path, sr=None)
            audio = apply_equalization(audio, sr)
            output_file_path = os.path.join(output_dir, filename)
            save_processed_audio(audio, output_file_path,sr)
# Main script logic

if __name__ == '__main__':
    answer = input('Do you want to reload data? ')
    if answer == 'yes':
        fire_repo = 'C:/Users/yassi/Desktop/projet iot 2'
        not_fire_repo = 'C:/Users/yassi/Desktop/projet iot 2'
        output_dir_fire = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/mic_data_fire'
        output_dir_not_fire = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/mic_data_not_fire'
        
        # Process and create databases for fire and not-fire data
        process_audio_files(fire_repo, output_dir_fire)
        #process_audio_files(not_fire_repo, output_dir_not_fire)
    else:
        print('Generation not authorised.')