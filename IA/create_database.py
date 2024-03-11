import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import datetime
import sys

def generate_spectr(sound_data_rep,file, show=False):
    # Load the WAV file
    sys.path.insert(0, sound_data_rep)
    sample_rate, audio_data = wav.read(file)
    
    # Check if audio data is multi-channel and mix down to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    nperseg = min(256, len(audio_data))

    # Compute the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate, nperseg=nperseg)
    min_freq = 0
    max_freq = 20000
    freq_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]

    spectrogram_cut = spectrogram[freq_indices, :]

    if show:
        plt.imshow(np.log1p(spectrogram_cut), aspect='auto', extent=[times.min(), times.max(), min_freq, max_freq])
        plt.colorbar(label='Log Normalized Spectrogram Amplitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Normalized Spectrogram')
        plt.show()

    return frequencies, times, spectrogram_cut
def write_head_csv(spectr_data_rep):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"spectrogram_data.csv")
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['frequency', 'time', 'spectrogram','label'])

def write_spectr_csv(spectr_data_rep, frequencies, times, spectrogram):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"spectrogram_data.csv")
    spectrogram_list = spectrogram
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data
        writer.writerow([
            frequencies.tolist(),  # Convert the numpy array to list and then to string
            times.tolist(),  # Convert the numpy array to list and then to string
            str(spectrogram_list),
            1,
            '\n'# The spectrogram is already a list of lists
        ])

    print(f"CSV file '{filename}' generated successfully.")

def create_database(sound_data_rep, spectr_data_rep):
    write_head_csv(spectr_data_rep)
    files = os.listdir(sound_data_rep)
    files_not_analized = []
    for file in files :
        print('Analyzing ' + file + '...')
        frequencies, times, spectrogram = generate_spectr(sound_data_rep,sound_data_rep+file)
        if len(times)>9800:
            print(file + ' eligible, spliting data...')
            for i in range(10): 
                write_spectr_csv(spectr_data_rep,frequencies, times[i*980:(i+1)*980], [[spectrogram[i][j] for i in range(len(spectrogram))] for j in range(i*980, (i+1)*980)])
        else :
            print(file + ' not eligible, skipping it')
            files_not_analized.append(file)
    
    if len(files_not_analized)!=0:
        print('Files not analyzed :')
        print(files_not_analized)
    else :
        print('All files analyzed successefully')


if __name__ == '__main__':
    sound_data_repo='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/'
    spectr_data_rep='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data'
    create_database(sound_data_repo, spectr_data_rep)