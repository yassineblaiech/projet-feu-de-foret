import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, concatenate
from tensorflow.keras.utils import to_categorical
import sys

# Load data, assuming CSV has 'time', 'frequency', 'spectrogram_data', 'label'
def load_data(csv_file):
    sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data")
    df = pd.read_csv(csv_file)
    # Example of how you might split your data into these components
    # This will need to be adjusted based on the actual structure of your CSV
    time_data = df['time'].apply(lambda x: np.array(x[1:-2].split(','), dtype=float)).tolist()
    frequency_data = df['frequency'].apply(lambda x: np.array(x[1:-2].split(','), dtype=float)).tolist()
    spectrogram_data = df.drop(['time', 'frequency', 'label'], axis=1).values
    labels = df['label'].values
    
    # Reshape spectrogram_data if necessary
    # Example reshape, adjust based on your data
    spectrogram_data=np.array(list(spectrogram_data))
    spectrogram_data = spectrogram_data.reshape(spectrogram_data.shape[0], len(frequency_data), len(time_data), 1)  # le shape c len(freq)*len(time)???
    
    return time_data, frequency_data, spectrogram_data, labels

# Define a more complex model that can take multiple inputs
def create_multi_input_model(spectrogram_shape, time_data_shape, frequency_data_shape):
    # Define input layers
    spectrogram_input = Input(shape=spectrogram_shape, name='spectrogram_input')
    time_input = Input(shape=(time_data_shape,), name='time_input')
    frequency_input = Input(shape=(frequency_data_shape,), name='frequency_input')

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
    
    model = create_multi_input_model(spectrogram_data[0].shape, len(time_data[0]), len(frequency_data[0]))
    # When fitting, provide a list of inputs corresponding to the model's inputs
    model.fit([X_train_spectrogram, X_train_time, X_train_frequency], y_train, batch_size=64, epochs=10, validation_split=0.2)
    
    # Evaluation would also need to be adjusted to provide multiple inputs
    score = model.evaluate([X_test_spectrogram, X_test_time, X_test_frequency], y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
if __name__=='__main__':
    sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data")
    time_data, frequency_data, spectrogram_data, labels=load_data('spectrogram_data.csv')
    print(time_data, frequency_data, spectrogram_data, labels)
