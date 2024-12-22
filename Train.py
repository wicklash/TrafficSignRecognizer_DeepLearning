import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score

# Step 1: Initialize lists to store data and labels
data = []
labels = []
classes = 43  # Number of classes (0-42)
cur_path = os.getcwd()  # Get the current working directory

# Step 2: Retrieve images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))  # Path to each class folder (0, 1, 2, ...)
    
    if os.path.exists(path):  # Check if directory exists
        images = os.listdir(path)
        
        for a in images:
            try:
                # Open image
                image = Image.open(os.path.join(path, a))
                image = image.resize((30, 30))  # Resize to 30x30
                image = np.array(image)  # Convert to numpy array
                data.append(image)  # Add image to the data list
                labels.append(i)  # Add label to the labels list
            except Exception as e:
                print(f"Error loading image {a}: {e}")
    else:
        print(f"Directory {path} does not exist.")

# Step 3: Convert lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# Step 4: Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}, Test labels shape: {y_test.shape}")

# Step 5: Convert the labels into one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

print("Data preprocessing complete.")

# Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# -- Part 4: Testing accuracy on the test dataset
y_test_csv = pd.read_csv('Test.csv')

labels = y_test_csv["ClassId"].values
imgs = y_test_csv["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

# Predict classes using the updated method
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

# Calculate accuracy with the test data
print(f"Test Accuracy: {accuracy_score(labels, pred_classes)}")
