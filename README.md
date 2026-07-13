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

### 💻 Run Locally

You can set this up two ways: with Git (recommended), or by downloading files manually if you don't have Git installed.

#### Option A: Using Git (recommended)

**1. Install Git LFS** (required — the model file is stored via Git LFS)

Download from [git-lfs.com](https://git-lfs.com), then run once:
```bash
git lfs install
```

**2. Clone this repository**

⚠️ Do **not** download the repo as a ZIP from GitHub — that only downloads a small placeholder for the model file, not the actual 33MB model. You must use `git clone` so Git LFS can pull the real file:
```bash
git clone https://github.com/Abdo0777/oral-disease-classification.git
cd oral-disease-classification
```

#### Option B: No Git — manual download

If you don't have Git installed, download each file individually instead (this avoids the ZIP/LFS placeholder issue entirely):

1. Create a new folder on your computer for the project.
2. Open each file below on GitHub, then click the **Download raw file** button (↓ icon) at the top right of the file view, and save it into that folder:
   - [app.py](https://github.com/Abdo0777/oral-disease-classification/blob/master/app.py)
   - [requirements.txt](https://github.com/Abdo0777/oral-disease-classification/blob/master/requirements.txt)
   - [class_indices.json](https://github.com/Abdo0777/oral-disease-classification/blob/master/class_indices.json)
   - [best_model_efficientnetb0_finetuned.keras](https://github.com/Abdo0777/oral-disease-classification/blob/master/best_model_efficientnetb0_finetuned.keras)
3. Make sure all four files end up in the same folder, with the model file around 33MB in size (if it's only a few hundred bytes, you downloaded the LFS placeholder by mistake — use the "Download raw file" button, not "Save page as").

#### Then, for either option:

**3. Install Python dependencies**

Requires Python 3.9+.
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

**5. Open the app**

The terminal will print a local URL, typically:
```
http://127.0.0.1:7860
```
Open that link in your browser, upload an oral image, and click **Analyze Image**.

**Troubleshooting**
- `ModuleNotFoundError`: make sure `pip install -r requirements.txt` completed without errors, and that you're running `python app.py` with the same Python installation `pip` installed packages into (if you have multiple Python versions, use the full path to that Python executable).
- **Windows with multiple Python installations**: if `pip install` succeeds but `python app.py` still throws `ModuleNotFoundError: No module named 'gradio'`, it usually means Windows is running a different Python than the one `pip` installed packages into (common if you have Python from python.org, MSYS2/MinGW, or the Microsoft Store installed at the same time). Check which Python `pip` is using:
  ```bash
  where pip
  ```
  This shows a path like `C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts\pip.exe`. Use the matching Python executable (same folder, just `python.exe` instead of `Scripts\pip.exe`) to run the app instead of plain `python`:
  ```bash
  C:\Users\<you>\AppData\Local\Programs\Python\Python313\python.exe app.py
  ```
- `File not found: best_model_efficientnetb0_finetuned.keras`: the model file wasn't downloaded correctly. With Git, re-clone with `git clone` (not a ZIP download). Manually, re-download using the "Download raw file" button and confirm it's ~33MB.

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
