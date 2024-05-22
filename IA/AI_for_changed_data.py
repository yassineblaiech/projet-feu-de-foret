import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Concatenate,Conv1D, MaxPooling1D, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import Callback
import sys

# Load data, assuming CSV has 'time', 'frequency', 'spectrogram_data', 'label'
def load_data(csv_file):
    with open(csv_file, mode='r', newline='') as file:
        lines=file.readlines()
        features=lines[0].strip('\n').split(',')
        dict_data={feature : [] for feature in features}
        for line in lines[1:]:
            data=line.strip('\n').split(';')
            for i in range(len(data)):
                list_data= [float(el) for el in data[i].split(',')]
                if len (list_data)>=150:
                    dict_data[features[i]].append(np.array(list_data).reshape(-1,150))
                else:
                    dict_data[features[i]].append(list_data[0])
        for feature in features:
            dict_data[feature]=np.array(dict_data[feature])
        print(features)
        
    return dict_data

# Define a more complex model that can take multiple inputs
def create_multi_input_model():
    # Main input (Mel Spectrogram)
    main_input = Input(shape=(128, 150, 1), name='main_input') # Assuming channel_last format

    # CNN layers for main input
    x = Conv2D(16, (3, 3), activation='relu')(main_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Additional inputs
    # Note: You might need to adjust the input shapes based on your preprocessing
    additional_inputs = []  # To keep track of all additional inputs
    for feature_name in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy']:
        input_layer = Input(shape=(1, 150), name=f'{feature_name}_input')
        additional_inputs.append(input_layer)
        # Example of processing additional inputs (reshape if necessary)
        y = Reshape((150,))(input_layer)  # Reshape for Dense layer
        y = Dense(64, activation='relu')(y)
        x = Concatenate()([x, y])

    # Spectral Contrast has a different shape
    spectral_contrast_input = Input(shape=(7, 150), name='spectral_contrast_input')
    additional_inputs.append(spectral_contrast_input)
    y = Conv1D(32, 3, activation='relu')(spectral_contrast_input)
    y = MaxPooling1D(2)(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    x = Concatenate()([x, y])

    # MFCCs
    mfccs_input = Input(shape=(13, 150), name='mfccs_input')
    additional_inputs.append(mfccs_input)
    y = Conv1D(32, 3, activation='relu')(mfccs_input)
    y = MaxPooling1D(2)(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    x = Concatenate()([x, y])
    
    # Tempo input
    tempo_input = Input(shape=(1,), name='tempo_input')  # Single value per sample
    additional_inputs.append(tempo_input)
    # Since it's already a single value, you might just concatenate directly after optional dense layers for transformation
    x = Concatenate()([x, tempo_input])

    # Final layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', name='output')(x)  # Assuming binary classification (fire/no fire)

    # Create model
    model = Model(inputs=[main_input, *additional_inputs], outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Updated training function to handle multiple inputs
def train_and_evaluate(csv_file):
    dict_data=load_data(csv_file)
    for feature in dict_data.keys():
        print(feature, 'has the shpe of ',dict_data[feature].shape)
    labels=dict_data['labels']
    dict_data = {feature: dict_data[feature] for feature in dict_data.keys() if feature != 'labels'}
    train_data, test_data, train_labels, test_labels = {}, {}, {}, {}
    for feature, data in dict_data.items():
        train_data[feature], test_data[feature]= train_test_split(
            data, test_size=0.2, random_state=42)
    train_labels['labels'],test_labels['labels']=train_test_split(
            labels, test_size=0.2, random_state=42)

    # Convert train_data and test_data from dict of arrays into list of arrays in the correct order
    train_data_list = [train_data[feature] for feature in dict_data.keys()]
    test_data_list = [test_data[feature] for feature in dict_data.keys()]
    train_labels_list=train_labels['labels']
    test_labels_list=test_labels['labels']
    # Assuming you have defined your model building function
    model = create_multi_input_model()  # Make sure this function is defined as discussed earlier

    # Define a custom callback to print accuracy after each epoch
    class AccuracyHistory(Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('accuracy'))
            print('Epoch:', len(self.acc), 'Accuracy:', logs.get('accuracy'))

    accuracy_history = AccuracyHistory()

    # Train the model
    history = model.fit(train_data_list, train_labels_list, validation_data=(test_data_list, test_labels_list),
                        epochs=10, callbacks=[accuracy_history])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data_list, test_labels_list)

    # Save the trained model
    model.save('my_trained_model_3.h5')
    print('Test accuracy:', test_accuracy)

    # Return the accuracy history
    return accuracy_history.acc

if __name__=='__main__':
    
    sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/mic_test.csv")
    train_and_evaluate('mic_test.csv')
    
    
        