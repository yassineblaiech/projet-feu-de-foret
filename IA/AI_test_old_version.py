import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import sys
import joblib
sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret")
from IA.AI_old_version import load_data
sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data")


def evaluate_model(model_path, csv_file):
    # Load the trained model
    model = load_model(model_path)
    
    # Load the scalers
    scaler_spectrogram = joblib.load('scaler_spectrogram.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
    scaler_frequency = joblib.load('scaler_frequency.pkl')
    # Load the test data
    time_data, frequency_data, spectrogram_data, labels = load_data(csv_file)
    X_test=scaler_spectrogram.transform(spectrogram_data.reshape(spectrogram_data.shape[0], -1)).reshape(spectrogram_data.shape),scaler_time.transform(time_data.reshape(time_data.shape[0], -1)).reshape(time_data.shape),scaler_frequency.transform(frequency_data.reshape(frequency_data.shape[0], -1)).reshape(frequency_data.shape)
    y_test=[int(e.strip('\r\n')) for e in labels]
    
    # Predict with the model; predictions are probabilities in [0, 1]
    predictions_prob = model.predict(X_test)
    print(predictions_prob)
    # Threshold predictions to get binary outcomes
    predictions = (predictions_prob > 0.5).astype(int)
    predictions = [0 if predictions[i][0] == 1 else 1 for i in range(len(predictions))]

    # Calculate F1 score; for binary classification, the default average is 'binary'
    f1 = f1_score(y_test, predictions)
    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print(y_test)
    print(predictions)
    
    print(f"Model F1 score on the test data: {f1}")
    print("Confusion Matrix:")
    print(cm)
    
    # X_train_focused = [X[misclassified_indices] for X in X_test]
    # Y_train_focused = y_test[misclassified_indices]
    # X_test_focussed=[]
    # y_test_focussed=[]
    # for i in range(len(X_train_focused)):
    #     X_train_focused[i],X_test_focussed[i]=train_test_split(
    #         X_train_focused[i], test_size=0.2, random_state=42)
    # Y_train_focused,y_test_focussed=train_test_split(Y_train_focused,test_size=0.2,random_state=42)
    # history = model.fit(X_train_focused, Y_train_focused, epochs=10, validation_data=(X_test_focussed, y_test_focussed))
    
    # # Calculate F1 score; for binary classification, the default average is 'binary'
    # f1 = f1_score(y_test, predictions)
    # # Generate confusion matrix
    # cm = confusion_matrix(y_test, predictions)
    
    # print(f"Model F1 score on the test data: {f1}")
    # print("Confusion Matrix:")
    # print(cm)
    
    return cm, f1

# Example usage:
if __name__ == '__main__':
    model_path = 'my_old_trained_model.h5'  # Update this path to where you saved your model
    test_csv_file = 'spectrogram_data_old_version_test.csv'  # Update this path to your test data CSV
    test_data=load_data(test_csv_file)
    cm,f1 = evaluate_model(model_path, test_csv_file)