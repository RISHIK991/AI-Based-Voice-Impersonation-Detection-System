import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, LSTM, TimeDistributed, Reshape,
                                     BatchNormalization, InputLayer)

# Set paths
train_path = r'D:\dataset\training'
val_path = r'D:\dataset\validation'
test_path = r'D:\dataset\testing'

# Parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 2  # real vs fake

# Utility to load RGB spectrograms and labels
def load_dataset(folder_path):
    data = []
    labels = []
    for label, subfolder in enumerate(['real', 'fake']):
        path = os.path.join(folder_path, subfolder)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    data.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    data, labels = np.array(data), np.array(labels)
    return shuffle(data, labels)

# Load datasets
print("Loading datasets...")
X_train, y_train = load_dataset(train_path)
X_val, y_val = load_dataset(val_path)
X_test, y_test = load_dataset(test_path)

# Normalize & one-hot encode
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Reshape for LSTM (convert image to sequence-like input: [batch, time, features])
TIME_STEPS = 28
FEATURES = IMG_WIDTH * 3  # RGB

def reshape_for_lstm(X):
    return X.reshape((X.shape[0], TIME_STEPS, -1))

X_train_seq = reshape_for_lstm(X_train)
X_val_seq = reshape_for_lstm(X_val)
X_test_seq = reshape_for_lstm(X_test)

# Build CNN+LSTM hybrid model
def build_cnn_lstm_model():
    model = Sequential()

    # CNN part
    model.add(InputLayer(input_shape=(TIME_STEPS, FEATURES)))
    model.add(Reshape((TIME_STEPS, IMG_WIDTH, 3)))  # Reshape to (time, width, channels)
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(Flatten()))

    # LSTM part
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_lstm_model()
model.summary()

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train
history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

# Evaluate
print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=1)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Classification Report
y_pred = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes, target_names=['Real', 'Fake']))