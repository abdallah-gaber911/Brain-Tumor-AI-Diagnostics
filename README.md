# Brain Tumor Classification (End-to-End Deep Learning Project)

A complete Machine Learning and Computer Vision project for classifying brain tumors from MRI scans into 4 distinct categories using PyTorch. This repository covers the full pipeline from raw data preprocessing to model training, evaluation, and deployment.

---

## 🚀 Project Overview
Brain tumor detection and classification from MRI images is a critical task in medical imaging. The goal of this project is to build an automated pipeline that can accurately classify MRI scans into four classes: **Glioma, Meningioma, Pituitary, and No Tumor**.

---

## 📊 1. Data Pipeline
* **Dataset Size:** Over **7,000 MRI images** balanced across all target classes.
* **Preprocessing & Augmentation:** 
  * Image resizing and normalization.
  * Applied data augmentation techniques (random rotations, flips, and scaling) to improve model generalization and prevent overfitting.
* **Data Splitting:** Divided into Training, Validation, and Test sets.

---

## 🧠 2. Model Architecture & Training
* **Framework:** PyTorch
* **Approach:** Utilized Convolutional Neural Networks (CNN) / Transfer Learning to extract robust spatial features from medical scans.
* **Training Details:** 
  * Optimized using Cross-Entropy Loss and Adam optimizer.
  * Implemented learning rate scheduling and early stopping based on validation metrics.

---

## 📈 3. Evaluation Metrics
The model was rigorously evaluated on unseen test data to ensure clinical reliability:
* **Test Accuracy:** **95%**
* **Recall / Sensitivity:** Exceeded **94%** across critical classes to minimize false negatives (ensuring no positive tumor cases are missed).
* **F1-Score:** Maintained a strong balance between precision and recall.

---

## 🌐 4. Deployment
* **Inference API / Interface:** Built a lightweight interactive interface/API where users can upload an MRI scan and get real-time predictions along with confidence scores in under a second.
* **Tech Stack:** Python, PyTorch, OpenCV, FastAPI / Streamlit (adjust based on your exact deployment method).

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
