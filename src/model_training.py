import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_model(data_path):
    df = pd.read_pickle(data_path)
    X = np.array(df['features'].tolist())
    y = np.array(df['label'].tolist())
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    np.save('models/label_encoder_classes.npy', le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    
    model.save('models/raga_model.keras')
    with open('models/model_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    train_model('data/processed_hindustani_data.pkl')
