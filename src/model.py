import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from preprocessing import load_data
from sklearn.model_selection import train_test_split

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def retrain_model(dataset_path):
    X, y = load_data(dataset_path)  # Load data from the provided path
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    model = load_model('models/pipe.pkl')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save('retrained_pipe.pkl')

if __name__ == "__main__":
    # Get the dataset path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python model.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]  # Path from command line argument
    retrain_model(dataset_path)