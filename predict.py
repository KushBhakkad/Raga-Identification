import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from src.feature_extraction import extract_features
import argparse

def predict_raaga(file_path, model_path, label_encoder_path):
    model = tf.keras.models.load_model(model_path)
    
    # Load the label encoder classes
    le_classes = np.load(label_encoder_path, allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = le_classes
    
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    
    return le.inverse_transform(predicted_class)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Raaga from audio file')
    parser.add_argument('file_path', type=str, help='Path to the audio file')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('label_encoder_path', type=str, help='Path to the label encoder classes file')
    
    args = parser.parse_args()
    
    raaga = predict_raaga(args.file_path, args.model_path, args.label_encoder_path)
    print(f'Predicted Raaga: {raaga}')
