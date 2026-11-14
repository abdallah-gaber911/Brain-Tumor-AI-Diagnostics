# 🧠 Brain Tumor MRI Classification & Risk Assessment

A Deep Learning application that classifies Brain MRI images into four categories using **ResNet18 (Transfer Learning)**. The project uses PyTorch for model training and Streamlit to provide an interactive web application for image classification and prediction visualization.

## 📌 Project Overview

Brain tumors are a serious medical condition that requires accurate examination and diagnosis.

This project aims to build an AI-powered image classification system that analyzes MRI images and predicts one of four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The application allows users to upload an MRI image and receive a predicted class, confidence score, and class probability visualization.

> **Disclaimer:** This is an educational machine learning project, not a clinically validated medical diagnostic system. Its predictions must not replace professional medical evaluation.

## 🎯 Project Objectives

- Apply Deep Learning to medical image classification.
- Use Transfer Learning with a pretrained ResNet18 model.
- Perform image preprocessing and data augmentation.
- Evaluate model performance using classification metrics.
- Build an interactive web application using Streamlit.
- Visualize prediction probabilities for all four classes.

## 🗂️ Dataset

**Brain Tumor MRI Dataset – Kaggle**

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains MRI images organized into four categories:

| Class | Description |
|---|---|
| Glioma | Glioma brain tumor |
| Meningioma | Meningioma brain tumor |
| Pituitary | Pituitary brain tumor |
| No Tumor | MRI images without a tumor |

Dataset structure:

```text
Brain-Tumor-MRI-Dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Dataset statistics:

| Dataset | Number of Images |
|---|---:|
| Training | 5,600 |
| Testing | 1,600 |
| Total | 7,200 |

## 🧠 Model Architecture

The project uses **ResNet18**, a Convolutional Neural Network (CNN) architecture based on residual learning.

### Transfer Learning

A pretrained ResNet18 model is used as the starting point. Its final fully connected layer is replaced with a new classification layer containing four output neurons.

```text
Input MRI Image
       |
       v
Image Preprocessing
       |
       v
ResNet18
       |
       v
Fully Connected Layer
       |
       v
4 Output Classes
       |
       v
Softmax Probabilities
       |
       v
Predicted Class
```

### Model Configuration

| Component | Configuration |
|---|---|
| Architecture | ResNet18 |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Image Normalization | ImageNet Mean and Standard Deviation |

## ⚙️ Data Preprocessing

All MRI images are resized to 224 × 224 pixels.

Training transformations:

- Resize to 224 × 224.
- Random Horizontal Flip.
- Random Rotation.
- Convert images to tensors.
- Normalize using ImageNet mean and standard deviation.

Validation and testing transformations:

- Resize to 224 × 224.
- Convert images to tensors.
- Normalize using ImageNet mean and standard deviation.

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The initial model evaluation achieved approximately **94% testing accuracy**.

Initial classification results:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 1.00 | 0.78 | 0.87 |
| Meningioma | 0.89 | 0.99 | 0.93 |
| No Tumor | 0.91 | 1.00 | 0.95 |
| Pituitary | 0.99 | 1.00 | 1.00 |
| **Overall Accuracy** | | **0.94** | |

*These metrics represent the initial model checkpoint. Results may differ after retraining or changing model configurations.*

## 💻 Web Application

The project includes an interactive web application built using Streamlit.

### Main Features

- Upload brain MRI images.
- Classify images into four categories.
- Display the predicted class.
- Show prediction confidence.
- Visualize probabilities for all classes.
- Display an AI-indicated tumor probability score.

### Application Workflow

```text
Upload MRI Image
       |
       v
Image Preprocessing
       |
       v
Trained ResNet18 Model
       |
       v
Classification
       |
       v
Prediction Results
       |
       v
Probability Visualization
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main Programming Language |
| PyTorch | Deep Learning Framework |
| Torchvision | Image Processing and ResNet18 |
| NumPy | Numerical Operations |
| Pandas | Data Handling |
| Matplotlib | Data Visualization |
| Scikit-learn | Model Evaluation |
| Streamlit | Web Application |
| PIL | Image Processing |

## 📁 Project Structure

```text
Brain-Tumor-MRI-Classification/
│
├── app.py
│
├── brain_tumor_model_v2.pth
│
├── requirements.txt
│
├── README.md
│
└── notebooks/
    └── Brain_Tumor_Classification.ipynb
```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Brain-Tumor-MRI-Classification
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📦 Requirements

The project requires:

```text
torch
torchvision
streamlit
numpy
pandas
matplotlib
scikit-learn
Pillow
```

## 🔮 Future Improvements

- Improve Glioma detection recall.
- Experiment with different CNN architectures.
- Apply Grad-CAM for model explainability.
- Improve model generalization on external MRI datasets.
- Add more comprehensive model evaluation.
- Explore model calibration and uncertainty estimation.
- Improve the user interface and deployment experience.

## 👨‍💻 Author

**Abdallah Mostafa Gaber**

Computer Science Student | Machine Learning & Deep Learning Enthusiast

---

⭐ If you find this project interesting, consider giving the repository a star!

**Note:** This project was developed for educational and learning purposes.
