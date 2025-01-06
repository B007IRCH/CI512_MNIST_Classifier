#MNIST Classifier 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import tkinter as tk
from tkinter import filedialog

# Configure API keys
os.environ["KAGGLE_USERNAME"] = "kylebirch"
os.environ["KAGGLE_KEY"] = "c1a2ecd4c8651eb69498103dc7fd144d"

# Function to handle zipped datasets
def handle_zipped_dataset(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The file '{zip_path}' does not exist.")
    extracted_folder = "extracted_data"
    os.makedirs(extracted_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
    csv_files = [f for f in os.listdir(extracted_folder) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the extracted dataset.")
    return os.path.join(extracted_folder, csv_files[0])

# Function to browse for a file
def browse_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("Zip files", "*.zip")]
    )
    if not file_path:
        raise ValueError("No file selected.")
    return file_path

# Load MNIST dataset
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return x_train, y_train, x_test, y_test

# Train neural network
def train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    return model, history

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    x_train, y_train, x_test, y_test = load_mnist_data()
    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape=(28 * 28,), num_classes=10)
    
    # Plot training results
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
