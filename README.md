# 🐄 Indian Cattle Breed Classifier

A fine-grained computer vision system that classifies **50 Indian cattle breeds** from photographs using **YOLO + ConvNeXtSmall**, served with a **FastAPI backend**, **Streamlit frontend**, and **Docker Compose containerization**.

---

## 🚀 Demo


![Streamlit Demo](images/streamlit_demo2.png)

---

## 🧠 Architecture

```text
User Image
    ↓
Streamlit Frontend
    ↓
FastAPI Backend
    ↓
YOLO Cattle Detection & Cropping
    ↓
ConvNeXtSmall Breed Classifier
    ↓
Top-3 Breed Predictions
```

The same YOLO-based cropping pipeline is used during both dataset preparation and inference, keeping preprocessing consistent and reducing background-related shortcuts.

---

## 📊 Results

**Held-out test accuracy: 52.72% across 50 breeds**

The held-out test set was built from a separate pool of farmer-recorded cattle videos, with **source-level (video-level) splitting** to prevent frame leakage between train, validation, and test sets. An earlier version of the pipeline had leakage between splits that inflated accuracy by more than 20 percentage points — this was diagnosed and fixed by rebuilding the splits at the video/source level rather than the frame level.

> **Why this number is lower than an earlier version of this project claimed (~75%), and why that's the right outcome:**
> The initial dataset merged two Kaggle datasets and used MD5 hashing to remove exact duplicates. But inspecting the model's decisions with Grad-CAM revealed the dataset still had significant noise — human faces, text, and busy background graphics — and the model was "cheating" (shortcut learning) by latching onto those background artifacts instead of the animal itself. The dataset was then strictly, manually cleaned to remove these misleading images, forcing the model to genuinely learn the correct features. That drops the accuracy score to 52.72% — lower, but honest, and far more representative of real-world performance.

**F1 Score by Breed**

![F1 scores per breed](images/f1_final.png)

**Grad-CAM**

Grad-CAM was used to verify that the model focuses on the cattle itself rather than background or contextual cues — fine-grained classifiers can otherwise hit good numbers while actually keying off background, lighting, or watermarks.

![Grad-CAM visualization](images/grad_cam_results.png)

*High-confidence, correctly-classified examples were specifically checked to rule out the model "getting lucky" via shortcuts — activation consistently lands on the animal rather than background clutter, though some diffuse (rather than tightly feature-localized) activation remains (see [Known Limitations](#known-limitations)).*

**MixUp Augmentation**

![MixUp example](images/images_after_mixup.png)

*MixUp blends pairs of training images and their labels proportionally, discouraging the model from memorizing sharp decision boundaries around individual training examples — a meaningful regularizer given several breed classes have fewer than 150 images.*

---

## 🗂️ Dataset

- 50 Indian cattle breeds
- Public datasets from Kaggle and other sources
- YouTube video frames collected for underrepresented breeds
- Perceptual hashing to remove near-duplicate images
- YOLO-based cattle cropping
- Manual data cleaning
- Source-level train/validation/test splitting to reduce video-frame leakage

The dataset is imbalanced, with roughly a **7.6x gap** between the largest and smallest classes — several rare breeds have fewer than 100 training images.

---

## 🏗️ Model

- **Backbone:** ConvNeXtSmall, pretrained on ImageNet
- **Input:** 224 × 224
- **Head:** Global Average Pooling → LayerNorm → Dense(256, GELU) → Dropout → 50-class Softmax
- **Training:** Progressive fine-tuning with staged backbone unfreezing, AdamW optimizer, class-weighted loss, MixUp augmentation to handle class imbalance
- **Precision:** Mixed float16

---

## 🔌 API

### `POST /predict`

Accepts an image and returns the predicted breed, confidence, and top-3 predictions.

**Example response:**

```json
{
  "predicted_breed": "Hariana",
  "confidence": 0.1916,
  "top_predictions": [
    {
      "breed": "Hariana",
      "confidence": 0.1916
    },
    {
      "breed": "Amritmahal",
      "confidence": 0.0928
    },
    {
      "breed": "Tharparkar",
      "confidence": 0.0805
    }
  ]
}
```

### `GET /health`

Checks whether the FastAPI service and both ML models (YOLO detector + classifier) are loaded and ready.

**FastAPI docs:** `http://localhost:8000/docs`

---

## 🐳 Docker

The application runs as two services:

- **FastAPI API**
- **Streamlit frontend**

Docker Compose uses an API health check so the frontend only starts once the backend and its models are fully loaded.

### Run

```bash
git clone https://github.com/Satvik524/cattle-breed-classification.git
cd cattle-breed-classification
docker compose up --build
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 📁 Repository Structure

```text
cattle-breed-classification/
├── api/
│   ├── Dockerfile
│   ├── inference.py
│   ├── main.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── user_input.py
├── images/
├── notebooks/
├── app.py
├── docker-compose.yml
├── Dockerfile.streamlit
├── requirements-api.txt
├── requirements-streamlit.txt
├── helper_functions.py
├── yolo26m.pt
└── README.md
```

---

## 🛠️ Tech Stack

- **Machine Learning:** TensorFlow, Keras, ConvNeXtSmall
- **Object Detection:** YOLO / Ultralytics
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Data Processing:** NumPy, Pandas, PIL, OpenCV, ImageHash
- **Deployment:** Docker, Docker Compose
- **Model Hosting:** Hugging Face Hub

---

## Known Limitations

- 52.72% accuracy on the current 50-breed held-out test set
- Performance varies significantly across breeds, especially rare classes
-- `Krishna_valley` currently scores 0 F1, primarily due to limited training data and the natural visual variation within the breed.
- Several other low-count classes also perform below the overall distribution.
- A handful of other low-count classes (under ~100 training images) score below the rest of the distribution for the same reason — fine-grained classification with this much per-class variation is a genuinely hard regime below a certain sample size
- When multiple cattle are detected in one image, only the highest-confidence detection is classified
- No explicit unknown/uncertain-breed rejection mechanism yet
- The held-out test set is relatively small, limiting the statistical strength of the evaluation

---

## 📌 Future Improvements

- Increase data for rare breeds to reduce the current ~7.6x class imbalance
- Improve confidence calibration and uncertainty handling
- Support multiple cattle in a single image
- Expand real-world evaluation data
- Optimize inference speed and Docker image size
- Deploy to AWS for public access
