# Brain Tumor Analysis System

# 🧠 Brain Tumor Analysis System

An AI-powered medical imaging application that combines MobileNetV2 classification, U-Net segmentation, Explainable AI (Grad-CAM), tumor size analysis, PDF report generation, and patient monitoring through an interactive Streamlit dashboard.

## 🚀 Features

### 🧠 Brain Tumor Classification
Uses a MobileNetV2 deep learning model to classify MRI scans into:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

### 🎯 U-Net Tumor Segmentation
Automatically detects and segments the tumor region from MRI images using a U-Net architecture.

### 📏 Tumor Size Analysis
Calculates:
- Tumor Area (%)
- Tumor Pixels
- Approximate Tumor Diameter
- Size Category (Small, Medium, Large)

### ⚠️ Severity Assessment
Determines tumor severity based on the segmented tumor area:
- Low
- Moderate
- High

### 📄 Medical Report Generation
Generates downloadable PDF reports containing:
- Tumor Type
- Prediction Confidence
- Tumor Area
- Severity Level

### 🔥 Explainable AI (Grad-CAM)
Visualizes the regions of the MRI scan that influenced the model's prediction, improving transparency and interpretability.

### 💾 Patient Database
Stores patient analysis records using SQLite for future reference and monitoring.

### 📋 Patient History Dashboard
Displays all previous patient records in a tabular format and supports CSV export.

### 📉 Tumor Progress Monitoring
Compares historical patient scans and visualizes tumor growth or reduction over time.

### 📊 Interactive Visualization Dashboard
Provides:
- Tumor Distribution Charts
- Segmentation Results
- Tumor Overlay Visualization
- Statistical Analysis

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Analysis-System.git
```

### 2. Navigate to Project Folder

```bash
cd Brain-Tumor-Analysis-System
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / Mac

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Project Structure

```text
Brain-Tumor-Analysis-System/
│
├── app.py
├── requirements.txt
├── models/
│   ├── classifier_model.keras
│   └── unet_model.keras
│
├── patient_history.db
└── README.md
```

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

### 8. Open in Browser

Streamlit will automatically open:

```text
http://localhost:8501
```

Upload an MRI image and explore:

- 🧠 Tumor Classification
- 🎯 U-Net Segmentation
- 📏 Tumor Size Analysis
- ⚠️ Severity Detection
- 📄 PDF Report Generation
- 🔥 Grad-CAM Explainable AI
- 💾 Patient Database
- 📋 Patient History Dashboard
- 📉 Tumor Progress Monitoring



## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- U-Net
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- SQLite
- FPDF
- Pandas