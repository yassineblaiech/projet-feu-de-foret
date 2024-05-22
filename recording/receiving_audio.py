import pyaudio
import wave
import socket


# Paramètres de l'audio
wav_output_filename='test.wav'
record_secs = 12 # seconds to record
form_1=pyaudio.paInt16
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
frames=[]

# Initialisation de PyAudio
p = pyaudio.PyAudio()

# Ouverture du flux de sortie
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

# Création du socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST = ''  # Adresse IP du récepteur (vide signifie toutes les interfaces disponibles)
PORT = 50007  # Port pour écouter
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connecté par', addr)

# Boucle de réception de l'audio
for ii in range(0,int((RATE/CHUNK)*record_secs)):
    data = conn.recv(CHUNK)
    frames.append(data)
    if not data:
        break
    stream.write(data)

# Fermeture des connexions
stream.stop_stream()
stream.close()
p.terminate()
conn.close()
s.close()
wavefile=wave.open(wav_output_filename,'wb')
wavefile.setnchannels(CHANNELS)
wavefile.setsampwidth(p.get_sample_size(form_1))
wavefile.setframerate(RATE)
wavefile.writeframes(b''.join(frames))
wavefile.close()
