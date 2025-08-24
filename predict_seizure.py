import pickle
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("seizure_model.h5")  # Load from .h5

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to preprocess input and make predictions
def predict_seizure(eeg_data):
    """
    Predict seizure probability based on EEG data.

    Args:
        eeg_data (list or numpy array): Raw EEG data for prediction.

    Returns:
        float: Probability of seizure (0 to 1).
    """
    eeg_data = np.array(eeg_data).reshape(1, -1)  # Reshape for model input
    scaled_data = scaler.transform(eeg_data)  # Apply the same scaling as training
    probability = model.predict(scaled_data)[0][0]  # Get probability of class 1 (seizure)
    return probability

