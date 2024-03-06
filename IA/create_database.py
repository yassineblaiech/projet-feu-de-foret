sound_data_repo = 'sound_data'
spectr_data_rep = 'spectr_data'

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt

def generate_spectr(file, show=True):
    # Load the WAV file
    sample_rate, audio_data = wav.read(file)

    # Set the number of FFT points for spectrogram calculation

    # Compute the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(audio_data, sample_rate)
    min_freq = 0
    max_freq = 20000

    # Find the indices corresponding to the frequency range
    freq_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]

    spectrogram_cut = spectrogram[freq_indices, :]

    # Normalize the spectrogram by its mean along the time axis
    normalized_spectrogram = spectrogram_cut / np.mean(spectrogram_cut, axis=1, keepdims=True)

    if show:
        # Display the normalized spectrogram
        plt.imshow(np.log1p(normalized_spectrogram), aspect='auto', extent=[times.min(), times.max(), min_freq, max_freq])
        plt.colorbar(label='Log Normalized Spectrogram Amplitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Normalized Spectrogram')
        plt.show()

    # Sum the normalized spectrogram along the frequencies axis
    magnitude_per_time = np.sum(spectrogram, axis=0)
    magnitude_normalized = magnitude_per_time / np.mean(magnitude_per_time)

    if show:
        plt.plot(times, magnitude_normalized)
        plt.xlabel('Time [sec]')
        plt.show()

    peak_value = max(magnitude_normalized)
    return peak_value
    if peak_value > 10:
        return 1
    else:
        return 0