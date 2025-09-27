import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



# ---------------- Seed ----------------
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# ---------------- Parameters ----------------
IMG_SIZE = (32, 32)
TIME_STEPS = 5  # number of consecutive frames in one sequence


# ---------------- Function to load images ----------------
def load_images(path, target_shape=IMG_SIZE):
    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = tf.keras.preprocessing.image.load_img(os.path.join(path, filename), target_size=target_shape)
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
    return np.array(images)

# ---------------- Paths ----------------
yawn_path = r'archive (79)\train\0'
no_yawn_path = r'archive (79)\train\1'

# ---------------- Load data ----------------
yawn_images = load_images(yawn_path)
no_yawn_images = load_images(no_yawn_path)


# ---------------- Normalize ----------------
yawn_images = yawn_images.astype('float32') / 255.0
no_yawn_images = no_yawn_images.astype('float32') / 255.0


# ---------------- Create sequences ----------------
def create_sequences(images, label, time_steps=TIME_STEPS):
    sequences = []
    labels = []
    for i in range(len(images) - time_steps + 1):
        seq = images[i:i+time_steps]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X_yawn, y_yawn = create_sequences(yawn_images, 0)
X_no_yawn, y_no_yawn = create_sequences(no_yawn_images, 1)


# ---------------- Combine data ----------------
X = np.concatenate([X_yawn, X_no_yawn], axis=0)
y_labels = np.concatenate([y_yawn, y_no_yawn], axis=0)


# ---------------- One-hot encode labels ----------------
y_labels = to_categorical(y_labels, num_classes=2)

# ---------------- Split train/test ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=seed_value, shuffle=True)


# ---------------- CNN-LSTM Model ----------------
model = Sequential()

# TimeDistributed CNN
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(TIME_STEPS, 32, 32, 3)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.3)))

model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.3)))

model.add(TimeDistributed(Flatten()))

# LSTM for temporal patterns
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.5))

# Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))


# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

# Plot training history (loss and accuracy)
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Plot the training history
plot_training_history(history)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")


#checking different evaluation metrics

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

y_pred=model.predict(X_test)


y_pred1_binary = (y_pred > 0.5).astype(int)

accuracy_score(y_test,y_pred1_binary)


y_test_classes = np.argmax(y_test, axis=1)       
y_pred_classes = np.argmax(y_pred, axis=1)       

print(f1_score(y_test_classes, y_pred_classes, average="weighted"))
print(precision_score(y_test_classes, y_pred_classes, average="weighted"))

# Save model in HDF5 format
model.save("yawn_model.h5")

# OR save in TensorFlow SavedModel format (folder)
# model.save("yawn_model")

