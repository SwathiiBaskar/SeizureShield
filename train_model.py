import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the EEG dataset
df = pd.read_csv('Epileptic Seizure Recognition.csv')

# Drop irrelevant columns
df = df.drop(columns=['Unnamed'], errors='ignore')  # Ignore errors if column doesn't exist

# Convert target variable ('y') to binary classification
df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)  # Seizure = 1, No seizure = 0

# Handle missing values
df = df.replace('-', np.nan)  # Replace hyphens with NaN
df = df.fillna(df.mean())  # Fill missing values with column mean

# Split into features and labels
X = df.drop(columns=['y']).values  # EEG signals (features)
y = df['y'].values  # Seizure labels (0 or 1)

# Standardize the input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Feedforward Neural Network (FNN) model
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Outputs a probability
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

# Save the trained model in HDF5 format
model.save("seizure_model.h5")

# Save the model in pickle format (if needed)
with open('seizure_model.pkl', 'wb') as file:
    pickle.dump(model, file)
