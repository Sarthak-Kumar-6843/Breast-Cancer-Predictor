# 🎗️ Breast Cancer AI Diagnostic Application

An interactive, clinical decision-support web application built with **Streamlit**, connected directly to a trained **Support Vector Machine (SVM)** classifier and **StandardScaler** pipeline for breast cancer malignancy prediction.

---

## 🚀 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
Run the following command from the project root directory:

```bash
streamlit run app.py
```

The web application will automatically open in your default browser at `http://localhost:8501`.

---

## 🌟 Application Features

### 1. 🔬 Single Patient Cytology Diagnostic
- **Organized Input Tabs**: The 30 Fine Needle Aspirate (FNA) cell nuclei morphometric measurements are partitioned logically into:
  1. **Mean Measurements** (Mean radius, texture, perimeter, area, smoothness, etc.)
  2. **Standard Error Measurements** (Variance/variability across biopsy nuclei)
  3. **Worst / Extreme Measurements** (Largest measurements, critical for malignancy detection)
- **Interactive Controls**: Dual slider and numerical precision inputs with biological tooltips.
- **Quick Clinical Presets**:
  - 🔴 **Malignant**: Loads the median morphometric profile of confirmed malignant cases.
  - 🟢 **Benign**: Loads the median morphometric profile of confirmed benign cases.
  - 🎲 **Random**: Samples an actual patient from the Wisconsin dataset.
  - 🔄 **Reset**: Returns inputs to overall population baseline averages.
- **Diagnostic Output**:
  - High-visibility status card (**Malignant Carcinoma Suspected** vs. **Benign Non-Cancerous**).
  - Calibrated model confidence score ($P \in [0\%, 100\%]$) derived from the SVM margin distance.
  - **Interactive Radar Chart**: Compares the 10 core nuclear features of the patient directly against benchmark Benign and Malignant population averages.
  - **Top Deviating Biomarkers**: Horizontal bar chart identifying which measurements deviate most significantly from healthy thresholds.

### 2. 📁 Batch CSV Predictor
- Upload a `.csv` file containing multiple patient cytology profiles.
- **Download Sample Batch CSV**: Includes a ready-to-test CSV template with 10 sample patients.
- Generates predictions, confidence scores, and margin distances for every row.
- Summary KPI indicators: Total cases screened, Malignant count & %, Benign count & %.
- Color-coded interactive results table.
- **Export**: One-click download of the complete annotated prediction report as a `.csv`.

### 3. 📈 Model Architecture & Validation Analytics
- Full transparency on model training and performance:
  - **Holdout Test Accuracy**: `98.25%`
  - **Recall (Sensitivity)**: `100.0%` (Zero false negatives on benign cases)
  - **Precision**: `97.26%`
  - **F1-Score**: `98.61%`
- Interactive **Confusion Matrix** heatmap (41 True Malignant, 71 True Benign, 2 False Benign, 0 False Malignant).
- Dataset class distribution pie chart (357 Benign vs 212 Malignant).
- Comparison table across models evaluated during exploration (SVM, Random Forest, XGBoost, Voting Ensemble, KNN, Logistic Regression, Naive Bayes).

### 4. 📖 Clinical Feature Guide
- Detailed medical definitions and diagnostic implications for each cytological measurement (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension).

---

## 🏗️ Model Architecture & Artifacts

| File | Description |
|---|---|
| `app.py` | Complete Streamlit web application with UI, charting, and inference engine. |
| `svm_model.pkl` | Trained scikit-learn `SVC` (Support Vector Classifier) with RBF kernel (`joblib`). |
| `scaler.pkl` | Fitted scikit-learn `StandardScaler` for the 30 FNA features (`joblib`). |
| `sample_batch_input.csv` | Ready-to-use CSV template containing 10 patient profiles for batch testing. |
| `feature_metadata.json` | Min, max, mean, and class-specific statistics for all 30 features. |
| `BreastCancerPrediction.ipynb` | Original data exploration, model training, and evaluation notebook. |
| `requirements.txt` | Python package dependencies. |

---

## ⚕️ Disclaimer
*This tool is intended for research, educational, and clinical decision-support purposes. All predictions should be interpreted by qualified medical practitioners in conjunction with definitive histopathological biopsy, mammography imaging, and clinical assessment.*

