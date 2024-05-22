import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import sys
sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret")
from IA.AI_for_a_large_data import load_data
sys.path.insert(0, "C:/Users/yassi/Desktop/projet iot 2/projet-feu-de-foret/IA/spectr_data")


def evaluate_model(model_path, csv_file):
    # Load the trained model
    model = load_model(model_path)
    
    # Load the test data
    data_dict = load_data(csv_file)
    # Assuming your data_dict structure, extract X_test and y_test
    X_test = [data_dict[key] for key in data_dict.keys() if key != 'labels']
    # Note: Depending on how load_data structures X_test, you might need to adjust this
    y_test = data_dict['labels']
    
    # Predict with the model; predictions are probabilities in [0, 1]
    predictions_prob = model.predict(X_test)
    # Threshold predictions to get binary outcomes
    predictions = (predictions_prob > 0.5).astype(int)
    misclassified_indices = np.where(np.round(predictions) != y_test)[0]
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
    model_path = 'my_trained_model_2.h5'  # Update this path to where you saved your model
    test_csv_file = 'mic_test_irl.csv'  # Update this path to your test data CSV
    test_data=load_data(test_csv_file)
    cm,f1 = evaluate_model(model_path, test_csv_file)