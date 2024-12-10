from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model(data_path, model_path):
    df = pd.read_pickle(data_path)
    X = np.array(df['features'].tolist())
    y = np.array(df['label'].tolist())
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    model = tf.keras.models.load_model(model_path)
    
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_encoded, y_pred_classes))

if __name__ == "__main__":
    evaluate_model('data/processed_hindustani_data.pkl', 'models/raga_model.keras')
