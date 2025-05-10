import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Step 1: Load and Preprocess Data
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    classes = os.listdir(data_dir)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode='grayscale')
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(class_name)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


data_dir = "C:\Image_Recog_and_Classification\dataset"
images, labels = load_data(data_dir)

# Convert labels to one-hot encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 2: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Step 4: Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Step 5: Save the Model
model.save('brain_tumor_model.keras')

# Step 6: Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model

# Load the model
model = load_model('brain_tumor_model.keras')

# Load and preprocess a new image
def preprocess_image(img_path, img_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

new_image = preprocess_image(r"C:\Image_Recog_and_Classification\image.jpeg")

prediction = model.predict(new_image)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print(f'Predicted Class: {predicted_class[0]}')
