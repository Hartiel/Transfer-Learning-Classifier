# 🐱 vs 🐶 Transfer Learning Classifier

A robust computer vision project implementing **Transfer Learning** with **MobileNetV2** to classify Cats and Dogs using the Microsoft Research dataset. 

This project was developed as part of the DIO Machine Learning Bootcamp.

## 🏗 Architecture & Tech Stack

* **Core:** Python, TensorFlow, Keras.
* **Model:** MobileNetV2 (Pre-trained on ImageNet) acting as a Feature Extractor.
* **Data Pipeline:** Custom ETL script for robust data ingestion and cleaning.
* **Environment:** Google Colab (GPU Accelerated).

## 🚀 Key Technical Implementation

Unlike standard tutorials, this implementation handles real-world dirty data issues found in the Microsoft dataset:

1.  **Robust ETL Pipeline:** The script includes a sanitation layer that detects and removes corrupt images (zero-byte files) before feeding the neural network, preventing runtime crashes.
2.  **Transfer Learning Strategy:** Utilized `MobileNetV2` with frozen weights (`include_top=False`, `trainable=False`) to leverage high-level feature extraction without the computational cost of training from scratch.
3.  **Regularization:** Applied `Dropout(0.2)` in the classification head to mitigate overfitting.

## 📊 Results

The model achieves **>95% validation accuracy** in just 3 epochs due to the power of transfer learning.

![Accuracy Graph]([INSERT_IMAGE_LINK_HERE])

## 📂 How to Run

1.  Open the notebook `transfer_learning_cats_dogs.ipynb` in Google Colab.
2.  Run all cells. The script automates the download of the dataset directly from Microsoft servers.

---
*Project developed by Arthur Reis.*
