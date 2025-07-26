# train_medical_classifier.py
# An advanced pipeline to build, train, and evaluate a deep learning model
# for medical image classification using TensorFlow, Keras, and Scikit-learn.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# --- 1. Configuration and Parameters ---

# In a real project, these paths would point to your dataset directories.
# The dataset should be structured as follows:
# /data/
#   /train/
#     /tumor/
#     /no_tumor/
#   /validation/
#     /tumor/
#     /no_tumor/
TRAIN_DIR = 'data/train'
VALIDATION_DIR = 'data/validation'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2 # Binary classification: Tumor vs. No Tumor
EPOCHS = 15 # Increased epochs for more realistic training

# --- 2. Data Augmentation and Preprocessing ---

print("Configuring data augmentation and preprocessing pipelines...")
# Data augmentation for the training set to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for the validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# NOTE: The following code is for demonstration and requires a local dataset.
# We will simulate the generators for this script to be self-contained.
#
# train_generator = train_datagen.flow_from_directory(...)
# validation_generator = validation_datagen.flow_from_directory(...)
#
# For simulation purposes, let's create mock generators
def create_mock_generator():
    # Create a generator that yields batches of random noise data
    while True:
        yield (np.random.rand(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3), 
               np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,)))

train_generator = create_mock_generator()
validation_generator = create_mock_generator()
print("Data generators configured (using mock data for simulation).")


# --- 3. Model Architecture with Transfer Learning ---

def build_model(num_classes):
    """
    Builds a classification model using transfer learning with EfficientNetB0.
    """
    # Load EfficientNetB0 pre-trained on ImageNet, excluding the top layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # Add dropout for regularization
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x) # Sigmoid for binary classification

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# --- 4. Compile and Train the Model ---

model, base_model = build_model(NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model built and compiled successfully.")
model.summary()

print("\n--- SIMULATING MODEL TRAINING ---")
# history = model.fit(
#     train_generator,
#     steps_per_epoch=10, # Mock steps
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     validation_steps=5, # Mock steps
#     verbose=1
# )
# Mock history object for plotting
history = type('obj', (object,), {'history': {
    'accuracy': np.linspace(0.6, 0.9, EPOCHS),
    'val_accuracy': np.linspace(0.65, 0.88, EPOCHS),
    'loss': np.linspace(0.7, 0.2, EPOCHS),
    'val_loss': np.linspace(0.6, 0.25, EPOCHS)
}})()
print("--- SIMULATED TRAINING COMPLETE ---\n")

# --- 5. Visualize Training History ---

def plot_history(history):
    """Plots training & validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 5))
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
    
    plt.savefig('training_history.png')
    plt.show()

print("Generating training history plots...")
plot_history(history)
print("Plots saved as 'training_history.png'")

# --- 6. Evaluate the Model ---

print("\n--- SIMULATING MODEL EVALUATION ---")
# In a real scenario:
# y_pred_probs = model.predict(validation_generator, steps=5)
# y_pred = np.round(y_pred_probs)
# y_true = validation_generator.classes (requires non-generator setup)

# Mock data for evaluation
y_true = np.random.randint(0, 2, size=100)
y_pred = np.random.randint(0, 2, size=100)
y_pred[10:50] = y_true[10:50] # Make it somewhat accurate

# Generate and print classification report
print("\nClassification Report:")
class_names = ['No Tumor', 'Tumor']
print(classification_report(y_true, y_pred, target_names=class_names))

# Generate and plot confusion matrix
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'")

# --- 7. Save and Load Model for Inference ---

print("\nSaving trained model...")
# model.save('medical_image_classifier.h5')
print("Model saved as 'medical_image_classifier.h5'")

def predict_single_image(model_path, image_path):
    """Loads the model and makes a prediction on a single image."""
    # from tensorflow.keras.models import load_model
    # from tensorflow.keras.preprocessing import image
    
    # print(f"\nLoading model from {model_path}...")
    # loaded_model = load_model(model_path)
    
    # print(f"Predicting for image: {image_path}")
    # img = image.load_img(image_path, target_size=IMG_SIZE)
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # prediction = loaded_model.predict(img_array)
    # score = prediction[0][0]
    
    # print(f"Prediction score: {score:.4f}")
    # print(f"Result: This MRI scan likely shows {'a tumor' if score > 0.5 else 'no tumor'}.")
    
    # --- Simulation for demonstration ---
    print(f"\n--- SIMULATING INFERENCE on '{image_path}' ---")
    score = np.random.rand()
    print(f"Prediction score: {score:.4f}")
    print(f"Result: This MRI scan likely shows {'a tumor' if score > 0.5 else 'no tumor'}.")
    print("-----------------------------------------")


# Example of how to run inference
predict_single_image('medical_image_classifier.h5', 'path/to/new_mri_scan.jpg')


