import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import sys
import joblib

# Load data, assuming CSV has 'time', 'frequency', 'spectrogram_data', 'label'
def load_data(csv_file):
    with open(csv_file, mode='r', newline='') as file:
        lines=file.readlines()[1:]
        spectrogram_data=[]
        frequency_data=[]
        time_data=[]
        labels=[]
        for line in lines:
            if line!='frequency,time,spectrogram,label\n':
                data=line.split(';')
                frequency_data.append([float(el) for el in data[0].split(',')[:117]])
                time_data.append([float(el) for el in data[1].split(',')])
                spectrogram_data.append(np.array([float(el) for el in data[2].split(',')]).reshape(99,980))
                labels.append(data[3])
        spectrogram_data=np.array(spectrogram_data)
        time_data=np.array(time_data)
        frequency_data=np.array(frequency_data)
        labels=np.array(labels)
        
        
    return time_data, frequency_data, spectrogram_data, labels

# Define a more complex model that can take multiple inputs
def create_multi_input_model(spectrogram_shape, time_data_shape, frequency_data_shape):
    # Define input layers
    spectrogram_input = Input(shape=(spectrogram_shape[0],spectrogram_shape[1],1), name='spectrogram_input')
    time_input = Input(shape=(time_data_shape[1],), name='time_input')
    frequency_input = Input(shape=(frequency_data_shape[1],), name='frequency_input')

    # Spectrogram processing branch
    x = Conv2D(32, (3, 3), activation='relu')(spectrogram_input)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Combining processed spectrogram with time and frequency data
    combined_inputs = concatenate([x, time_input, frequency_input])

    # Fully connected layers
    y = Dense(64, activation='relu')(combined_inputs)
    output = Dense(2, activation='softmax')(y)
    
    # Creating and compiling the model
    model = Model(inputs=[spectrogram_input, time_input, frequency_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Updated training function to handle multiple inputs
def train_and_evaluate(csv_file):
    time_data, frequency_data, spectrogram_data, labels = load_data(csv_file)
    # Preprocessing, normalization, etc., goes here
    
    # Prepare data for model input, ensure shapes are correct
    # Convert labels to categorical
    labels = to_categorical(labels)
    
    # Split the data
    # You'll need to ensure this split applies correctly across all your inputs and labels
    X_train_spectrogram, X_test_spectrogram, y_train, y_test = train_test_split(spectrogram_data, labels, test_size=0.2, random_state=42)
    X_train_time, X_test_time, y_train, y_test = train_test_split(time_data, labels, test_size=0.2, random_state=42)
    X_train_frequency, X_test_frequency, y_train, y_test = train_test_split(frequency_data, labels, test_size=0.2, random_state=42)
    # Similarly split time_data, frequency_data
    # Apply scalers to the data
    scaler_spectrogram = StandardScaler().fit(X_train_spectrogram.reshape(X_train_spectrogram.shape[0], -1))
    X_train_spectrogram_scaled = scaler_spectrogram.transform(X_train_spectrogram.reshape(X_train_spectrogram.shape[0], -1)).reshape(X_train_spectrogram.shape)

    scaler_time = StandardScaler().fit(X_train_time.reshape(X_train_time.shape[0], -1))
    X_train_time_scaled = scaler_time.transform(X_train_time.reshape(X_train_time.shape[0], -1)).reshape(X_train_time.shape)

    scaler_frequency = StandardScaler().fit(X_train_frequency.reshape(X_train_frequency.shape[0], -1))
    X_train_frequency_scaled = scaler_frequency.transform(X_train_frequency.reshape(X_train_frequency.shape[0], -1)).reshape(X_train_frequency.shape)
    joblib.dump(scaler_spectrogram, 'scaler_spectrogram.pkl')
    joblib.dump(scaler_time, 'scaler_time.pkl')
    joblib.dump(scaler_frequency, 'scaler_frequency.pkl')
    # Transform the test data using the same scalers
    X_test_spectrogram_scaled = scaler_spectrogram.transform(X_test_spectrogram.reshape(X_test_spectrogram.shape[0], -1)).reshape(X_test_spectrogram.shape)
    X_test_time_scaled = scaler_time.transform(X_test_time.reshape(X_test_time.shape[0], -1)).reshape(X_test_time.shape)
    X_test_frequency_scaled = scaler_frequency.transform(X_test_frequency.reshape(X_test_frequency.shape[0], -1)).reshape(X_test_frequency.shape)

    model = create_multi_input_model(spectrogram_data[0].shape,( 1,len(time_data[0])), (1,len(frequency_data[0])))
    # When fitting, provide a list of inputs corresponding to the model's inputs
    model.fit([X_train_spectrogram, X_train_time, X_train_frequency], y_train, batch_size=64, epochs=30, validation_split=0.2)
    
    # Save the trained model
    model.save('my_old_trained_model.h5')
    
    # Evaluation would also need to be adjusted to provide multiple inputs
    score = model.evaluate([X_test_spectrogram, X_test_time, X_test_frequency], y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
if __name__=='__main__':
    sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data")
    time_data, frequency_data, spectrogram_data, labels=load_data('spectrogram_data_old_version.csv')
    spectrogram_shape, time_data_shape, frequency_data_shape=spectrogram_data.shape,time_data.shape,frequency_data.shape
    create_multi_input_model(spectrogram_shape, time_data_shape, frequency_data_shape)
    train_and_evaluate('spectrogram_data_old_version.csv')