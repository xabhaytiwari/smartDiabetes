import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""Enhanced Version of Recurrent Neural Networks"""
"""RNN Challenges"""
# 1. Vanishing Gradient
# 2. Exploding Gradient

"""LSTM Cell"""
# 1. Forget Gate
# 2. Input Gate
# 3. Output Gate

def bareBones_LSTM_Model(file_path):
    df = pd.read_csv(file_path)

    #Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    #Normalise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Reshape
    X_lstm = X_scaled.reshape((X_scaled.shape[0]), 1, X_scaled.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs = 60, batch_size=32, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


              