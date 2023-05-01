import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

def predict_future(model, features):
    predictions = []
    last_window = features[-1:].shape(1, features.shape[1], 1)
    for i in range(30):
        prediction = model.predict(last_window)[0][0]
        predictions.append(prediction)
        last_window = np.append(last_window[:,1:,:],[[prediction]],axis=1)
    return predictions
