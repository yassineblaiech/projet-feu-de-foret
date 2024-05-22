import csv
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import datetime
import sys
from other_features import extract_audio_features
from other_features import feature_names


def write_head_csv(spectr_data_rep):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"mic_test_irl.csv")
    with open(filename, mode='w', newline='') as file:
        feature_names_list= feature_names()
        for feature in feature_names_list:
            file.write("{},".format(feature))
        file.write('labels\n')

def write_spectr_fire_csv(spectr_data_rep, features,i):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"mic_test_irl.csv")
    str_features={}
    with open(filename, mode='a') as file:
        # Convert the arrays to comma-separated strings
        for key in features.keys():
            if key in ['spectrogram', 'spectral_contrast_input','mfccs_input']:
                str_features[key] =','.join(','.join(map(str, row)) for row in [[round(features[key][i][j],2) for i in range(len(features[key]))] for j in range(i*400, (i+1)*400)])
            elif key in ['tempo_input']:
                str_features[key] =str(features[key])
            else:
                str_features[key] =','.join(map(str, list(round(el,2) for el in features[key][0][i*400:(i+1)*400])))
        
        # Write the strings to the file directly, avoiding CSV writer to prevent quotes
        for key in features.keys():
            file.write(f"{str_features[key]};")
        file.write("1\n")
    print(f"CSV file '{filename}' part {i} generated successfully.")
    
def write_spectr_not_fire_csv(spectr_data_rep, features,i):
    sys.path.insert(0, spectr_data_rep)
    filename = os.path.join(spectr_data_rep, f"mic_test_irl.csv")
    str_features={}
    with open(filename, mode='a') as file:
        # Convert the arrays to comma-separated strings
        for key in features.keys():
            if key in ['spectrogram', 'spectral_contrast_input','mfccs_input']:
                str_features[key] =','.join(','.join(map(str, row)) for row in [[round(features[key][i][j],2) for i in range(len(features[key]))] for j in range(i*400, (i+1)*400)])
            elif key in ['tempo_input']:
                str_features[key] =str(features[key])
            else:
                str_features[key] =','.join(map(str, list(round(el,2) for el in features[key][0][i*400:(i+1)*400])))
        
        # Write the strings to the file directly, avoiding CSV writer to prevent quotes
        for key in features.keys():
            file.write(f"{str_features[key]};")
        file.write("0\n")
    print(f"CSV file '{filename}' part {i} generated successfully.")
    
def create_database_fire(sound_data_fire_rep, spectr_data_rep):
    write_head_csv(spectr_data_rep)
    files = os.listdir(sound_data_fire_rep)
    files_not_analized = []
    for file in files :
        print('Analyzing ' + file + '...')
        features = extract_audio_features(sound_data_fire_rep+file)
        times=features['spectrogram'].shape[1]
        if times>400:
            print(file + ' eligible, spliting data...')
            for i in range(times//400): 
                write_spectr_fire_csv(spectr_data_rep,features,i)
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
        features = extract_audio_features(sound_data_not_fire_rep+file)
        times=features['spectrogram'].shape[1]
        if times>400:
            print(file + ' eligible, spliting data with times={}...'.format(times))
            for i in range(times//400):
                write_spectr_not_fire_csv(spectr_data_rep,features,i)
        else :
            print(file + 'skipping with times={}...'.format(times))
            files_not_analized.append(file)
    
    if len(files_not_analized)!=0:
        print('Files not analyzed :')
        print(files_not_analized)
    else :
        print('All files analyzed successefully')



if __name__ == '__main__':
    answer=input('do you want to reload data?')
    if answer=='yes':
        sound_data_fire_repo='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/mic_data_fire/'
        sound_data_not_fire_repo='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/sound_data/mic_data_not_fire/'
        spectr_data_rep='C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data'
        create_database_fire(sound_data_fire_repo,spectr_data_rep)
        create_database_not_fire(sound_data_not_fire_repo, spectr_data_rep)
    else:
        print('generation not authorised')