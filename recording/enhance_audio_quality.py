import librosa
import numpy as np
import soundfile as sf
import pydub

# Load and save functions to facilitate file handling
def load_audio(audio_path):
    return librosa.load(audio_path, sr=None)

def save_audio(audio_path, y, sr):
    sf.write(audio_path, y, sr)

# 1. Volume Normalization
def normalize_volume(y, target_dBFS=-20.0):
    audio = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)
    change_in_dBFS = target_dBFS - audio.dBFS
    return np.array(audio.apply_gain(change_in_dBFS).get_array_of_samples())

# 2. Dynamic Range Compression
# Note: This function is a placeholder, real compression would require a dedicated library
def compress_dynamic_range(y, threshold=-20, ratio=4):
    return (np.sign(y) * (np.log1p(ratio*np.abs(y/np.exp(threshold))) / np.log1p(ratio)))

# 3. Equalization - Focuses on emphasizing frequencies important for fire crackling sounds
def apply_equalization(y, sr):
    D = librosa.stft(y)
    hz_per_bin = sr / D.shape[0]
    # Assuming fire crackles are prominent around 2000 to 5000 Hz
    low_bin = int(200 / hz_per_bin)
    high_bin = int(500 / hz_per_bin)
    D[low_bin:high_bin, :] *= 20  
    return librosa.istft(D)

# 4. Noise Reduction - A basic spectral gating approach
def reduce_noise(y, sr):
    D = librosa.stft(y)
    magnitude = np.abs(D)
    threshold = np.median(magnitude)
    D[magnitude < threshold] = 0
    return librosa.istft(D)

# 5. Spectral Enhancement - Emphasizing higher frequencies for crackling sounds
def spectral_enhancement(y, sr):
    D = librosa.stft(y)
    hz_per_bin = sr / D.shape[0]
    # Boost higher frequencies more, assuming they are more indicative of crackles
    enhancement_curve = np.linspace(1.0, 2.0, int(D.shape[0]/2))
    D[:int(D.shape[0]/2), :] *= enhancement_curve[:, None]
    return librosa.istft(D)

# Example usage
audio_path = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/test3.wav'
y, sr = load_audio(audio_path)

# Apply each function
#y_norm = normalize_volume(y)
#y_comp = compress_dynamic_range(y)
y_eq = apply_equalization(y, sr)
#y_nr = reduce_noise(y_eq, sr)
#y_enhanced = spectral_enhancement(y_nr, sr)

# Save the final enhanced audio
enhanced_audio_path = 'C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/test3_enhanced.wav'
save_audio(enhanced_audio_path, y_eq, sr)