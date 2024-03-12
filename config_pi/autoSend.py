import requests as rq
import random
import time
import wave
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal 
import pyaudio


FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100
CHUNK=1024
RECORD_SECONDS=5


url="http://foret.uber.space/status?id=0&status="


min_freq=13000
max_freq=20000

def is_fire(sample_rate,audio_data):
	frequencies,times,spectrogram=signal.spectrogram(audio_data,sample_rate)

	freq_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]

	spectrogram_cut=spectrogram[freq_indices,:]

	magnitude_per_time = np.sum(spectrogram_cut, axis=0)

	mmean=np.mean(magnitude_per_time)
	normalized_magnitude=[0]
	if not mmean==0:
		normalized_magnitude= magnitude_per_time/mmean

	peak_value = max(normalized_magnitude)
	assert peak_value > 0
	print(peak_value)

	if 100 > peak_value > 10:
		return True
	else:
		return False





INTERVAL=2

def scan_ip():
    i=0
    try:
        audio=pyaudio.PyAudio()
        stream=audio.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
        while True:
            print("LOOPING...")
            frames=[]
            for _ in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
                data=stream.read(CHUNK)
                frames.append(data)
            output_file=f"/home/username/Desktop/SOUNDS/test{i}.wav"
            print("Saving file to "+output_file)
            with wave.open(output_file,"wb") as wavefile:
                wavefile.setnchannels(CHANNELS)
                wavefile.setsampwidth(audio.get_sample_size(FORMAT))
                wavefile.setframerate(RATE)
                wavefile.writeframes(b"".join(frames))
            sample_rate, sound_data = wav.read(output_file)
            print("Testing for a fire : ")
            if is_fire(sample_rate, sound_data):
                print("KO")
                rq.get(url+"ko")
            else:
                print("OK")
                rq.get(url+"ok")
            i+=1
        print("Closing")
    except:
        print("error")
        

if __name__ == '__main__':
    print("Starting program")
    while(True):    
        scan_ip()
        time.sleep(1)
