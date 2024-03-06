sound_data_repo = 'sound_data'
spectr_data_rep = 'spectr_data'

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import datetime

def generate_spectr(file, show=False):
    # Load the WAV file
    sample_rate, audio_data = wav.read(file)

    # Set the number of FFT points for spectrogram calculation
    nperseg = min(256, len(audio_data))  # Choose a smaller value if input signal is shorter than 256

    # Compute the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate, nperseg=nperseg)
    min_freq = 0
    max_freq = 20000

    # Find the indices corresponding to the frequency range
    freq_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]

    spectrogram_cut = spectrogram[freq_indices, :]

    if show:
        # Display the normalized spectrogram
        plt.imshow(np.log1p(spectrogram_cut), aspect='auto', extent=[times.min(), times.max(), min_freq, max_freq])
        plt.colorbar(label='Log Normalized Spectrogram Amplitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Normalized Spectrogram')
        plt.show()

    return frequencies, times, spectrogram_cut


def write_spectr(spectr_data_rep, frequencies, times, spectrogram):
    filename = os.path.join(spectr_data_rep, f"spectrogram_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    # Write data to the file
    with open(filename, 'w') as file:
        file.write(np.array2string(frequencies, separator=',') + "\n\n")
        file.write(np.array2string(times, separator=',') + "\n\n")
        for i in range(len(spectrogram)):
            file.write(np.array2string(spectrogram[i], separator=',') + "\n")
            
        #file.write(np.array2string(spectrogram[2], separator=',') + "\n")
    print(f"File '{filename}' generated successfully.")


if __name__ == '__main__':
    frequencies, times, spectrogram = generate_spectr('sound1.wav')
    #write_spectr('..\\spectr_data',frequencies, times, spectrogram)
