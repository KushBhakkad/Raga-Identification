import os
import pandas as pd
from src import data_preprocessing, model_training, model_evaluation, prediction

def main():
    # Step 1: Preprocess data
    hindustani_dir = 'data/hindustani/'
    
    print("Preprocessing data...")
    hindustani_df = data_preprocessing.preprocess_data(hindustani_dir)
    
    hindustani_df = data_preprocessing.augment_data(hindustani_df)
    hindustani_df.to_pickle('data/processed_hindustani_data.pkl')
    
    # Step 2: Train the model
    print("Training the model...")
    model_training.train_model('data/processed_hindustani_data.pkl')
    
    # Step 3: Evaluate the model
    print("Evaluating the model...")
    model_evaluation.evaluate_model('data/processed_hindustani_data.pkl', 'models/raga_model.keras')
    
    # Step 4: Predict raga for a new audio file
    print("Predicting raga for a new audio file...")
    file_path = 'data/new_audio/yaman26.wav'
    raaga = prediction.predict_raaga(file_path, 'models/raga_model.keras', 'models/label_encoder_classes.npy')
    print(f'Predicted Raaga: {raaga}')

if __name__ == "__main__":
    main()
