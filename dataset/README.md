# Dataset Information

This project uses the **Fruits Fresh and Rotten for Classification** dataset for training and evaluating the deep learning model.

---

## 📌 Dataset Source
- **Platform:** Kaggle  
- **Dataset Name:** Fruits Fresh and Rotten for Classification  
- **Link:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification  

---

## 📂 Expected Dataset Structure
dataset/
├── train/
│ ├── fresh/
│ └── rotten/
└── test/
├── fresh/
└── rotten/


---

## ⚠️ Important Note
Due to GitHub file size limitations, the full dataset is **not uploaded** to this repository.

To run the project:
1. Download the dataset from Kaggle
2. Extract it
3. Place the `train` and `test` folders inside the `dataset/` directory

---

## 🧠 Usage
The dataset is used for:
- Training the CNN model
- Validating performance
- Testing accuracy and freshness detection

The images are resized to **240 × 240 pixels** and normalized before being fed into the model.
