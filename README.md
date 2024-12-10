# Raga-Identification

Raga-Identification is a machine learning-based application designed to predict the raga of a given audio file. It leverages feature extraction, a pre-trained deep learning model, and label encoding to identify ragas in Indian classical music.

**Features**
- Accepts audio files as input.
- Extracts relevant features for raga identification.
- Utilizes a pre-trained TensorFlow model for prediction.
- Supports multiple ragas defined in the dataset.
- Provides user-friendly command-line interaction.

**Dataset**
- Saraga: research datasets of Indian Art Music
- Data Source: https://zenodo.org/records/4301737

**Requirements**
- Python 3.8 or 3.9
- Tensorflow 2.8.0 and Keras 2.8.0 for comaptibility

**Command to Run:**
- python3 predict.py <audio_file_path> <model_path> <label_encoder_path>
- Example: python3 predict.py data/new_audio/yaman26.wav models/raga_model.h5 models/label_encoder_classes.npy

**Acknowledgements**
- TensorFlow for the deep learning framework.
- Librosa for audio processing.
- Indian classical music community for their support and inspiration.

**Output**

![Output](https://github.com/user-attachments/assets/7b7d28ea-9d2e-4d3c-a750-aea964498909)
