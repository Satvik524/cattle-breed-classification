# cattle-breed-classification
Classifying 50 Indian cattle breeds using ConvNeXt-Tiny.
# 🐄 Indian Cattle Breed Classification (50 Classes)

This project implements a Deep Learning pipeline using **TensorFlow** to classify 50 different breeds of Indian cattle. The model utilizes transfer learning to achieve robust performance across a diverse 50-class distribution.

## 🚀 Technical Overview
* **Framework:** TensorFlow / Keras
* **Dataset:** 50 Classes, processed with MD5-based deduplication to ensure data integrity.
* **Architecture:** Modern Convolutional Neural Network (CNN) architecture optimized for multi-class classification.
* **Environment:** Developed in Google Colab using a T4 GPU.

## 📊 Results
The model's performance was evaluated based on per-class F1-scores, which provide a balanced view of precision and recall across all 50 breeds.

![F1 Score Analysis](F1_score.png)

## 🛠️ How to Use
1. Open the `.ipynb` notebook in Google Colab.
2. Ensure the dataset is accessible in the specified directory or mounted via Google Drive.
3. Execute the preprocessing cells to handle data merging and deduplication.
4. Run the training cells or load the saved model weights for inference.
