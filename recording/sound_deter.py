import librosa
import soundfile as sf
import numpy as np

# Charger un enregistrement de feu de forêt
file_path = "C:\\Users\\thoma\\Desktop\\2A\\feux de forêts\\Projet_Detecteur_Feu\\projet\\projet-feu-de-foret\\IA\\sound_data\\sound1.wav"
audio, sr = librosa.load(file_path, sr=None)

# Dégrader la qualité audio
def degrade_audio(audio,sr,target_sr, noise_level=0.01):
    # Rééchantillonnage
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Ajout de bruit
    noise = np.random.normal(0, noise_level, len(audio))
    degraded_audio = audio + noise
    
    # Normalisation
    degraded_audio /= np.max(np.abs(degraded_audio))
    
    return degraded_audio

# Dégrader l'audio
target_sr = 16000  # Votre fréquence d'échantillonnage cible
degraded_audio = degrade_audio(audio, target_sr)

# Enregistrer l'audio dégradé
output_path = "C:\\Users\\thoma\\Desktop\\2A\\feux de forêts\\Projet_Detecteur_Feu\\projet\\projet-feu-de-foret\\IA\\sound_data\\sound_deter16k.wav"
sf.write(output_path, degraded_audio, target_sr)

