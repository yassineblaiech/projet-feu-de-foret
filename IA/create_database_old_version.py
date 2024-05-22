import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import datetime
import sys

def generate_spectr(sound_data_rep,file, show=True):
    # Load the WAV file
    sys.path.insert(0, sound_data_rep)
    sample_rate, audio_data = wav.read(file)
    
    # Check if audio data is multi-channel and mix down to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    nperseg = min(256, len(audio_data))

    # Compute the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate, nperseg=nperseg)
    frequencies=np.array([round(num, 2) for num in frequencies])[:982]
    times=[round(num,2) for num in times][:982]
    spectrogram=np.array([[round(num,2) for num in row] for row in spectrogram])[:982]
    min_freq = 5000
    max_freq = 25000
    freq_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]
    times_array=np.array(times)
    spectrogram_cut = spectrogram[freq_indices, :]

    if show:
        plt.imshow(np.log1p(spectrogram_cut), aspect='auto', extent=[times_array.min(), times_array.max(), min_freq, max_freq])
        plt.colorbar(label='Log Normalized Spectrogram Amplitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Normalized Spectrogram')
        plt.show()

    return frequencies, times, spectrogram_cut

def write_head_csv(spectr_data_rep):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"spectrogram_data_test.csv")
    with open(filename, mode='w', newline='') as file:
        file.write('frequency,time,spectrogram,label\n')

def write_spectr_fire_csv(spectr_data_rep, frequencies, times, spectrogram):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"spectrogram_data_test.csv")
    with open(filename, mode='a') as file:
        # Convert the arrays to comma-separated strings
        frequencies_str = ','.join(map(str, frequencies))
        times_str = ','.join(map(str, times))
        spectrogram_str = ','.join(','.join(map(str, row)) for row in spectrogram)
        
        # Write the strings to the file directly, avoiding CSV writer to prevent quotes
        file.write(f"{frequencies_str};{times_str};{spectrogram_str};1\n")
    print(f"CSV file '{filename}' generated successfully.")
    
def write_spectr_not_fire_csv(spectr_data_rep, frequencies, times, spectrogram):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"spectrogram_data_test.csv")
    with open(filename, mode='a') as file:
        # Convert the arrays to comma-separated strings
        frequencies_str = ','.join(map(str, frequencies))
        times_str = ','.join(map(str, times))
        spectrogram_str = ','.join(','.join(map(str, row)) for row in spectrogram)
        
        # Write the strings to the file directly, avoiding CSV writer to prevent quotes
        file.write(f"{frequencies_str};{times_str};{spectrogram_str};0\n")
    print(f"CSV file '{filename}' generated successfully.")

def create_database_fire(sound_data_fire_rep, spectr_data_rep):
    write_head_csv(spectr_data_rep)
    files = os.listdir(sound_data_fire_rep)
    files_not_analized = []
    for file in files :
        print('Analyzing ' + file + '...')
        frequencies, times, spectrogram = generate_spectr(sound_data_fire_rep,sound_data_fire_rep+file)
        if len(times)>980:
            print(file + ' eligible, spliting data...')
            for i in range(len(times)//980): 
                write_spectr_fire_csv(spectr_data_rep,frequencies, times[i*980:(i+1)*980], [[spectrogram[i][j] for i in range(len(spectrogram))] for j in range(i*980, (i+1)*980)])
        else :
            print(file + ' not eligible, skipping it')
            files_not_analized.append(file)
    
    if len(files_not_analized)!=0:
        print('Files not analyzed :')
        print(files_not_analized)
    else :
        print('All files analyzed successefully')
        
def create_database_not_fire(sound_data_not_fire_rep, spectr_data_rep):

    files = os.listdir(sound_data_not_fire_rep)
    files_not_analized = []
    for file in files :
        print('Analyzing ' + file + '...')
        frequencies, times, spectrogram = generate_spectr(sound_data_not_fire_rep,sound_data_not_fire_rep+file)
        if len(times)>980:
            print(file + ' eligible, spliting data...')
            for i in range(len(times)//980): 
                write_spectr_not_fire_csv(spectr_data_rep,frequencies, times[i*980:(i+1)*980], [[spectrogram[i][j] for i in range(len(spectrogram))] for j in range(i*980, (i+1)*980)])
        else :
            print(file + ' not eligible, skipping it')
            print(len(times))
            files_not_analized.append(file)
    
    if len(files_not_analized)!=0:
        print('Files not analyzed :')
        print(files_not_analized)
    else :
        print('All files analyzed successefully')


if __name__ == '__main__':
    answer=input('do ypu want to reload data?')
    if answer=='yes':
        sound_data_fire_repo='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/fire_test/'
        sound_data_not_fire_repo='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/not_fire_test/'
        spectr_data_rep='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data'
        create_database_fire(sound_data_fire_repo,spectr_data_rep)
        create_database_not_fire(sound_data_not_fire_repo, spectr_data_rep)
    else:
        print('generation not authorised')