# MINI-PROJECT-5
# 🐟 Multiclass Fish Image Classification using Deep Learning & Streamlit

This project focuses on classifying multiple species of fish using deep learning.  
Five state-of-the-art CNN models were trained and compared:

- **VGG16**
- **ResNet50**
- **MobileNetV2**
- **NASNetMobile**
- **EfficientNetB0**

The model with the best validation accuracy was selected and saved as  
`best_fish_model.h5`.

A **Streamlit Web App** is built to allow users to upload fish images and see:

- Top 5 predicted classes  
- Probability scores  
- Final predicted fish species  

---

## 📂 Project Features

✔ Multi-model training & comparison  
✔ Automatic model selection  
✔ Streamlit-based interactive UI  
✔ Image preprocessing & augmentation  
✔ GPU-accelerated training (Colab)  
✔ Real-time prediction with probability visualization  

---

## 🧠 Model Workflow

1. Load and preprocess dataset  
2. Train 5 CNN models  
3. Compare validation accuracies  
4. Save best-performing model  
5. Load model in Streamlit  
6. Predict top-5 fish categories  

---

## 📦 Dataset Structure

Your dataset should follow this format:


Uploading images

Predicting fish category

Displaying model confidence scores

Dataset/
│── train/
│ ├── Black_Sea_Sprat/
│ ├── Gilt_Head_Bream/
│ ├── Hourse_Mackerel/
│ ├── Red_Mullet/
│ ├── Sea_Bass/
│ ├── Striped_Red_Mullet/
│ ├── Trout/
│ ├── ...
│
│── val/
│ ├── same class folders as train/


---

## 🖼 Streamlit App

Upload an image → get predictions like:

📊 Prediction Results (Top 5):
Black_Sea_Sprat: 100.00%
Gilt_Head_Bream: 0.00%
Striped_Red_Mullet: 0.00%
Sea_Bass: 0.00%
Trout: 0.00%

✅ Predicted Fish: Black_Sea_Sprat
