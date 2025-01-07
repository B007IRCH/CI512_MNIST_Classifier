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
import tkinter as tk
from tkinter import filedialog
import kagglehub

# Configure API keys for Kaggle
os.environ["KAGGLE_USERNAME"] = "kylebirch"
os.environ["KAGGLE_KEY"] = "c1a2ecd4c8651eb69498103dc7fd144d"

# Load Fashion MNIST dataset from Kaggle
def load_fashion_mnist():
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    print(f"Fashion MNIST dataset downloaded to: {path}")

    # Load training and test data
    train_data = pd.read_csv(os.path.join(path, "fashion-mnist_train.csv"))
    test_data = pd.read_csv(os.path.join(path, "fashion-mnist_test.csv"))

    # Extract features and labels
    x_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values

    # Reshape data
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

# Load user-provided dataset
def load_custom_dataset():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV Dataset",
        filetypes=[("CSV files", "*.csv")]
    )

    if not file_path:
        raise ValueError("No file selected.")

    data = pd.read_csv(file_path)

    # Assume last column is the label
    x = data.iloc[:, :-1].values / 255.0
    y = data.iloc[:, -1].values

    # Reshape if needed
    x = x.reshape(-1, 28 * 28)

    # One-hot encode labels
    y = to_categorical(y, num_classes=10)

    # Split into training and testing (80-20 split)
    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

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

# Plot results
def plot_results(history):
    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

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
    print("Choose dataset:")
    print("1. Use Fashion MNIST dataset (Kaggle)")
    print("2. Upload your own dataset")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        x_train, y_train, x_test, y_test = load_fashion_mnist()
    elif choice == '2':
        x_train, y_train, x_test, y_test = load_custom_dataset()
    else:
        print("Invalid choice. Exiting.")
        return

    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape=(28 * 28,), num_classes=10)

    # Plot results
    plot_results(history)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
