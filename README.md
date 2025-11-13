# MINI-PROJECT-5
# Multiclass Fish Image Classification
📘 Project Overview

The Multiclass Fish Image Classification project focuses on classifying images of fish into multiple species using Deep Learning. It involves training a CNN model from scratch and applying Transfer Learning with pre-trained architectures to achieve higher accuracy. The project is deployed as a Streamlit web app that allows users to upload images and receive instant predictions.

🧠 Skills Demonstrated

Deep Learning

Python

TensorFlow / Keras

Data Preprocessing & Augmentation

Transfer Learning

Model Evaluation & Visualization

Streamlit Deployment

Model Comparison & Saving

🎯 Problem Statement

To develop an automated system that classifies fish images into predefined categories. The goal is to build an accurate and efficient classification model using both custom and pre-trained CNN architectures.

💼 Business Use Cases

Enhanced Accuracy – Identify the best-performing model for fish classification.

Deployment Ready – Deliver a user-friendly Streamlit web app for real-time fish recognition.

Model Comparison – Evaluate and compare models to select the optimal approach for deployment.

⚙️ Approach
1. Data Preprocessing and Augmentation

Rescaled images to the [0, 1] range.

Applied augmentation techniques like rotation, zoom, and horizontal flips.

2. Model Training

Trained a CNN from scratch.

Fine-tuned five pre-trained models:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

Saved the best-performing model (.h5 or .pkl) for deployment.

3. Model Evaluation

Compared metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

Visualized training history (accuracy & loss) for each model.

4. Deployment

Built a Streamlit web app for:

Uploading images

Predicting fish category

Displaying model confidence scores
