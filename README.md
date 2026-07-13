# 🦷 Oral Diseases Image Classification

An end-to-end deep learning system that classifies oral disease images into 6 categories: **Calculus, Caries, Gingivitis, Hypodontia, Tooth Discoloration, and Ulcers**. The project covers data preparation, custom CNN development, transfer learning, hyperparameter tuning, comparative analysis, and deployment as an interactive web app.

## 📊 Dataset

- **Source:** [Oral Diseases Dataset](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) (Kaggle)
- **Total images:** 11,653
- **Classes:** 6 (Calculus, Caries, Gingivitis, Hypodontia, Tooth Discoloration, Ulcers)
- **Split:** 70% train / 15% validation / 15% test (stratified)

## 🧠 Models Trained

| Model | Approach | Test Accuracy | Test Loss |
|---|---|---|---|
| Custom CNN | Built from scratch (4 conv blocks) | 90.22% | 0.2394 |
| ResNet50 | Transfer learning (frozen base) | 90.16% | 0.2308 |
| EfficientNetB0 | Transfer learning (frozen base) | 91.19% | 0.1954 |
| **EfficientNetB0 (fine-tuned)** | Transfer learning + fine-tuning top 30 layers | **91.30%** | 0.2136 |

**Best model:** EfficientNetB0, fine-tuned — selected for deployment.

## 🏗️ Architecture Details

**Custom CNN:**
- 4 Conv2D blocks (32 → 64 → 128 → 256 filters) with BatchNormalization and MaxPooling
- GlobalAveragePooling2D + Dense(256) + Dropout(0.5) head
- ~457K parameters, trained fully from scratch

**Pretrained Models (ResNet50 / EfficientNetB0):**
- ImageNet-pretrained base, initially frozen
- Custom classification head: GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(6, softmax)
- Each model uses its own `preprocess_input` function (critical for correct performance)

**Fine-tuning:**
- Unfroze the top 30 layers of EfficientNetB0's base
- Retrained with a low learning rate (1e-5) to adapt pretrained features to the domain

## ⚙️ Hyperparameter Tuning

- Optimizer: Adam
- Learning rate scheduling via `ReduceLROnPlateau`
- Early stopping on validation loss (`patience=5`, best weights restored)
- Fine-tuning learning rate reduced to 1e-5 to avoid destroying pretrained weights
- Data augmentation: rotation, shift, zoom, horizontal flip, brightness variation (train set only)

## 📈 Comparative Analysis

All four models performed within a narrow band (90.16%–91.30%), indicating the dataset is well-balanced and learnable even without transfer learning. EfficientNetB0 outperformed ResNet50 despite having far fewer trainable parameters, due to its more efficient compound scaling architecture. Fine-tuning provided a modest but real improvement over the frozen version.

## 🚀 Deployment

The best-performing model (EfficientNetB0, fine-tuned) is deployed via **Gradio**, allowing users to upload an oral image and receive real-time classification with confidence scores across all 6 classes.

### Run locally
```bash
pip install -r requirements.txt
python app.py
```

## 📁 Project Structure
```
├── notebook.ipynb                              # Full training pipeline
├── best_model_efficientnetb0_finetuned.keras   # Saved best model
├── class_indices.json                          # Label mapping
├── app.py                                      # Gradio deployment app
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- TensorFlow / Keras
- ResNet50, EfficientNetB0 (ImageNet pretrained)
- Gradio (deployment)
- scikit-learn, pandas, matplotlib, seaborn

## 📌 Requirements
```
tensorflow
gradio
numpy
pandas
pillow
scikit-learn
matplotlib
seaborn
```

## 🚀 Try It Live
👉 [**Click here to try the classifier**](https://huggingface.co/spaces/Abdo495/oral-disease-classifier)
