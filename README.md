# Fetal Ultrasound Plane Classifier

A deep learning–based web application for classifying fetal ultrasound images into anatomical planes using **ResNet-50 / EfficientNet** models and deployed with **Streamlit**.

---

##  Project Overview

This project uses convolutional neural networks to classify fetal ultrasound images into the following categories:

* Fetal abdomen
* Fetal brain
* Fetal femur
* Fetal thorax
* Maternal cervix
* Other

The application provides real-time predictions via an interactive Streamlit interface.

---

##  Models Used

*  ResNet-50 (Best performing model)
*  EfficientNet-B0 (Alternative model)

 Final deployed model:

```
fetal_classifier_final.pth
```

---

##  Project Structure

```
fetal-ultrasound-classifier/
│
├── app.py                         # Streamlit app
├── fetal_classifier_final.pth     # Final trained model
├── best_resnet50.pth              # ResNet checkpoint
├── best_efficientnet.pth          # EfficientNet checkpoint
├── fetal_fine_tunning.ipynb       # Training notebook
│
├── data/
│   └── fetal_data/Images          # Dataset images
│
├── output_image/                  # Generated outputs
│
└── FETAL_PLANES_ZENODO.zip        # Original dataset
```

---

##  Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/fetal-ultrasound-classifier.git
cd fetal-ultrasound-classifier
```

---

### 2️⃣ Install dependencies

```
pip install streamlit torch torchvision pillow matplotlib numpy
```

---

##  Run the Streamlit App

```
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

##  How to Use

1. Upload a fetal ultrasound image (`.png`, `.jpg`, `.jpeg`)
2. The model predicts the anatomical plane
3. View:

   * Predicted class
   * Confidence score
   * Probability distribution

---

##  Important Notes

* Ensure the model file exists in the same directory:

```
fetal_classifier_final.pth
```

* The app uses **ResNet-50 architecture** for loading the model.

* If you face model loading errors:

  * Ensure correct architecture (ResNet vs EfficientNet)

---

##  Features

* ✔ Real-time image classification
* ✔ Confidence score visualization
* ✔ Class probability chart
* ✔ Clean Streamlit UI

---

##  Dataset

Dataset used:

**FETAL_PLANES_DB (Zenodo)**

* Contains labeled fetal ultrasound images
* Multiple anatomical plane categories

---

##  Future Improvements

*  Grad-CAM visualization (model explainability)
*  Ensemble models (ResNet + EfficientNet)
*  Cloud deployment (Streamlit Cloud)
*  Mobile-friendly UI

---

##  Acknowledgements

* PyTorch
* Streamlit
* Zenodo Dataset

