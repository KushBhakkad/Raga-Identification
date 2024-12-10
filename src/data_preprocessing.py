import os
import pandas as pd
from src.feature_extraction import extract_features
from sklearn.utils import shuffle
import numpy as np

def preprocess_data(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp3') and not file.startswith("._"):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)  # Use the immediate parent directory name as the label
                features = extract_features(file_path)
                if features is not None and len(features) > 0:
                    data.append([features, label])
                    print(f"Processed {file}: {features[:5]}...")  # Print first 5 feature values for debug
                else:
                    print(f"Failed to extract features from {file}")
    df = pd.DataFrame(data, columns=['features', 'label'])
    return df

def augment_data(df):
    augmented_data = []
    for i, row in df.iterrows():
        features, label = row['features'], row['label']
        noise = np.random.randn(len(features)) * 0.005
        augmented_data.append([features + noise, label])
        augmented_data.append([features * (1 + np.random.uniform(-0.1, 0.1)), label])
    augmented_df = pd.DataFrame(augmented_data, columns=['features', 'label'])
    return pd.concat([df, augmented_df], ignore_index=True)

if __name__ == "__main__":
    hindustani_dir = 'data/hindustani/'
    
    print("Preprocessing Hindustani music data...")
    hindustani_df = preprocess_data(hindustani_dir)
    
    hindustani_df = shuffle(hindustani_df)
    hindustani_df = augment_data(hindustani_df)
    hindustani_df.to_pickle('data/processed_hindustani_data.pkl')
