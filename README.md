# AI-Based Fruit Classification and Freshness Detection Using Deep Learning 🍎🍌🍊

This project implements an AI-based system to automatically classify fruits and detect their freshness using Deep Learning techniques. The system uses a Convolutional Neural Network (CNN) trained on fruit images to identify fruit types and determine whether they are fresh or rotten.

---

## 📌 Problem Statement
Manual fruit inspection in agriculture and retail is time-consuming, subjective, and error-prone. This project aims to automate fruit classification and freshness detection using image-based deep learning, thereby improving accuracy, efficiency, and reducing food waste.

---

## 🎯 Objectives
- Automatically classify fruit types from images  
- Detect freshness levels (Fresh / Rotten)  
- Reduce dependency on manual inspection  
- Provide a scalable AI-based solution for real-world applications  

---

## 🧠 Technology Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Image Processing:** OpenCV, PIL  
- **Visualization:** Matplotlib  
- **Dataset Source:** Kaggle  
- **Training Platform:** Google Colab (GPU)

---


## 🏗️ Model Architecture
The CNN model consists of:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling layers
- Dropout layers to prevent overfitting
- Fully connected Dense layers
- Softmax output layer for multi-class classification

**Input Size:** 240 × 240 × 3  
**Total Parameters:** ~12.9 million  

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/AI-Fruit-Freshness-Detection.git
cd AI-Fruit-Freshness-Detection
```
### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Download Dataset

Download dataset from Kaggle

Place it as:

dataset/
 ├── train/
 └── test/

---

## 📊 Results

- High accuracy achieved during training and validation

- Stable loss convergence

- Effective detection of fresh vs rotten fruits

- Confusion matrix and classification report used for evaluation

- Sample outputs include:

- Accuracy & loss plots

- Test image predictions

- Confusion matrix visualization

---

## 📸 Sample Outputs

- Training accuracy and loss graphs
 [Model Accuracy Graph](Result-images/Model-accuracy.png) , [Model Loss Graph](Result-images/Model-loss.png) 

- Sample predictions on test images
  [Prediction Image](Result-images/Test-prediction-image.png) 

- Confusion matrix for performance evaluation
  [Confusion Matrix](Result-images/confusion-matrix.png) 
