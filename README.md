Advanced Medical Image Classification Pipeline

This project demonstrates a complete, end-to-end deep learning pipeline for classifying medical images (e.g., MRI scans of brains to detect tumors). It is built using TensorFlow and Keras and showcases advanced techniques including transfer learning with modern architectures, data augmentation, and comprehensive model evaluation.

This repository represents a professional-grade approach to solving a real-world computer vision problem.
Methodology & Key Techniques

    Transfer Learning: Instead of training a model from scratch, this project leverages transfer learning. It uses the EfficientNetB0 model, pre-trained on the vast ImageNet dataset, as a powerful feature extractor. This approach significantly reduces training time and improves accuracy, especially with limited medical imaging data.

    Data Augmentation: To prevent overfitting and make the model more robust, the training dataset is artificially expanded using data augmentation. The ImageDataGenerator applies random transformations—such as rotations, shifts, zooms, and flips—to the training images in real-time.

    Custom Classification Head: The top classification layers of the pre-trained EfficientNetB0 are removed and replaced with a custom "head" consisting of GlobalAveragePooling2D, Dropout for regularization, and Dense layers, culminating in a final sigmoid activation function for binary classification.

    Comprehensive Evaluation: The model's performance is not just measured by accuracy. The pipeline generates a detailed Classification Report (including precision, recall, and F1-score) and a Confusion Matrix to provide a deeper insight into its predictive capabilities on different classes.

Project Structure

This repository contains a single, comprehensive Python script:

    train_medical_classifier.py: This script contains the entire pipeline:

        Configuration: Sets up parameters like image size, batch size, and file paths.

        Data Preprocessing: Configures ImageDataGenerator for both training (with augmentation) and validation.

        Model Building: Defines the build_model function that constructs the transfer learning architecture.

        Training: Compiles and (simulates) trains the model.

        Visualization: Generates and saves plots for training/validation accuracy and loss.

        Evaluation: Generates and saves a classification report and a confusion matrix.

        Inference: Includes a function to demonstrate how to load the saved model and make a prediction on a new, unseen image.

How to Run
Prerequisites

You will need a Python environment with the following libraries installed:

pip install tensorflow matplotlib scikit-learn seaborn

Setup

    Create the Dataset Directory: The script expects a specific folder structure. Create a data directory with train and validation subdirectories. Inside each, create folders for your classes (e.g., tumor, no_tumor).

    /data/
      /train/
        /tumor/
          - mri_tumor_01.jpg
          - ...
        /no_tumor/
          - mri_normal_01.jpg
          - ...
      /validation/
        /tumor/
          - ...
        /no_tumor/
          - ...

    Update the Script: If your class names are different, update the class_names list in the evaluation section of the script.

Execution

    Run the training script from your terminal:

    python train_medical_classifier.py

    The script will simulate the training process and generate three output files in the same directory:

        training_history.png: A plot of the model's accuracy and loss over epochs.

        confusion_matrix.png: A heatmap visualizing the model's prediction accuracy.

        medical_image_classifier.h5: The saved, trained model file (simulated).
