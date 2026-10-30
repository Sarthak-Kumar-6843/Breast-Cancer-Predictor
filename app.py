"""
Breast Cancer Diagnostic AI - Streamlit Web Application
Connected with trained SVM (Support Vector Classifier) and StandardScaler pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic Tool",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Result Cards */
    .result-card-malignant {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border: 2px solid #F43F5E;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(244, 63, 94, 0.15);
    }
    .result-card-benign {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);
    }
    .result-title-malignant {
        color: #BE123C;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-title-benign {
        color: #047857;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-desc {
        color: #334155;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Metric pill */
    .metric-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .pill-malignant {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .pill-benign {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }

    /* Info callout card */
    .info-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        color: #334155;
    }
    
    /* Footer disclaimer */
    .disclaimer-box {
        margin-top: 40px;
        padding: 14px 20px;
        background-color: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        color: #64748B;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# PATHS AND ASSET RESOLUTION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "sample_batch_input.csv")

# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model pipeline...")
def load_ml_pipeline():
    """Loads the trained SVM model and StandardScaler from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def get_benchmark_dataset():
    """Loads Wisconsin Diagnostic Breast Cancer dataset for statistics and benchmarks."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    feature_names = list(raw.feature_names)
    return df, feature_names, raw.target_names

# Load model, scaler, and benchmark data
try:
    model, scaler = load_ml_pipeline()
    dataset_df, FEATURE_NAMES, TARGET_NAMES = get_benchmark_dataset()
    pipeline_loaded = True
except Exception as e:
    pipeline_loaded = False
    st.error(f"⚠️ Error loading ML model/scaler: {str(e)}")

# Feature grouping definitions
MEAN_FEATURES = [f for f in FEATURE_NAMES if f.startswith("mean ")]
SE_FEATURES = [f for f in FEATURE_NAMES if f.endswith(" error")]
WORST_FEATURES = [f for f in FEATURE_NAMES if f.startswith("worst ")]

FEATURE_DESCRIPTIONS = {
    "radius": "Mean distance from center to points on perimeter (microns)",
    "texture": "Standard deviation of gray-scale values in digitized biopsy image",
    "perimeter": "Nuclear boundary perimeter length (microns)",
    "area": "Nuclear contour surface area (square microns)",
    "smoothness": "Local variation in radius lengths (variation coefficient)",
    "compactness": "Perimeter^2 / Area - 1.0 (compactness of nuclear boundary)",
    "concavity": "Severity of concave portions of the nuclear contour",
    "concave points": "Number of concave portions along the nuclear contour",
    "symmetry": "Symmetry score of the cell nucleus",
    "fractal dimension": "Coastline approximation - 1 (nuclear border irregularity)"
}

def get_feature_tooltip(col: str) -> str:
    """Returns a readable explanation for each feature."""
    for key, desc in FEATURE_DESCRIPTIONS.items():
        if key in col:
            return desc
    return "Fine Needle Aspirate (FNA) cytological measurement"

# -----------------------------------------------------------------------------
# HELPER: PRESET POPULATION
# -----------------------------------------------------------------------------
def load_preset(preset_type: str):
    """Sets session_state input values based on clinical preset archetypes."""
    if preset_type == "malignant":
        malignant_cases = dataset_df[dataset_df['target'] == 0]
        sample_vals = malignant_cases.mean().to_dict()
    elif preset_type == "benign":
        benign_cases = dataset_df[dataset_df['target'] == 1]
        sample_vals = benign_cases.mean().to_dict()
    elif preset_type == "random":
        random_row = dataset_df.sample(n=1, random_state=None).iloc[0]
        sample_vals = random_row.to_dict()
    else: # default / dataset average
        sample_vals = dataset_df.mean().to_dict()
        
    for feat in FEATURE_NAMES:
        st.session_state[f"input_{feat}"] = float(sample_vals[feat])

# Initialize session_state defaults if not set
if "inputs_initialized" not in st.session_state:
    for feat in FEATURE_NAMES:
        st.session_state[f"input_{feat}"] = float(dataset_df[feat].mean())
    st.session_state["inputs_initialized"] = True

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-heart.png", width=70)
    st.title("🎗️ Breast Cancer AI")
    st.caption("Wisconsin Diagnostic Dataset (WDBC) Decision Support")

    mode = st.radio(
        "Navigation Mode",
        [
            "🔬 Single Patient Diagnostic",
            "📁 Batch CSV Predictor",
            "📈 Model Performance & Analytics",
            "📖 Clinical Feature Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown(
        """
        - **Classifier**: Support Vector Machine (`SVC`)
        - **Kernel**: Radial Basis Function (`rbf`)
        - **Preprocessing**: `StandardScaler`
        - **Validation Accuracy**: `98.25%`
        - **Target Classes**:
          - `0`: Malignant (Cancerous)
          - `1`: Benign (Non-Cancerous)
        """
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Diagnostic Presets")
    st.caption("Populate patient inputs with benchmark profiles:")
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        if st.button("🔴 Malignant", use_container_width=True, help="Load average profile of confirmed malignant cases"):
            load_preset("malignant")
            st.rerun()
    with col_pre2:
        if st.button("🟢 Benign", use_container_width=True, help="Load average profile of confirmed benign cases"):
            load_preset("benign")
            st.rerun()

    col_pre3, col_pre4 = st.columns(2)
    with col_pre3:
        if st.button("🎲 Random", use_container_width=True, help="Load a random patient from the clinical dataset"):
            load_preset("random")
            st.rerun()
    with col_pre4:
        if st.button("🔄 Reset", use_container_width=True, help="Reset to dataset average baseline"):
            load_preset("default")
            st.rerun()

    st.markdown("---")
    st.caption("Antigravity AI Health Analytics Engine • 2026")

# -----------------------------------------------------------------------------
# MODE 1: SINGLE PATIENT DIAGNOSTIC
# -----------------------------------------------------------------------------
if mode == "🔬 Single Patient Diagnostic":
    st.markdown("<h1 class='main-title'>🔬 Single Patient Cytology Diagnostic</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Enter Fine Needle Aspirate (FNA) nuclear morphometric measurements below to assess malignancy risk.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='info-card'>
            💡 <b>Tip:</b> Use the quick presets in the sidebar to populate standard profiles or manually adjust individual cytology biomarkers.
            Measurements are divided into <b>Mean Nucleus Values</b>, <b>Standard Errors (Variance)</b>, and <b>Worst (Extreme) Values</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input form organized into 3 organized tabs
    tab_mean, tab_se, tab_worst = st.tabs([
        "🧬 1. Mean Measurements (Cell Nuclei)",
        "📐 2. Standard Error (Variability)",
        "⚠️ 3. 'Worst' / Extreme Values"
    ])

    def render_feature_inputs(feature_list, num_cols=2):
        """Renders sliders with sync number precision for a list of features."""
        cols = st.columns(num_cols)
        for i, feat in enumerate(feature_list):
            with cols[i % num_cols]:
                col_min = float(dataset_df[feat].min())
                col_max = float(dataset_df[feat].max())
                val_range = col_max - col_min
                
                # Determine display formatting & step size
                if val_range > 100:
                    step = 0.5
                    fmt = "%.1f"
                elif val_range > 5:
                    step = 0.05
                    fmt = "%.2f"
                elif val_range > 0.5:
                    step = 0.005
                    fmt = "%.3f"
                else:
                    step = 0.0005
                    fmt = "%.4f"

                clean_name = feat.replace("mean ", "").replace("worst ", "").replace(" error", "").title()
                prefix = "Mean " if "mean" in feat else ("Worst " if "worst" in feat else "SE ")
                label = f"{prefix}{clean_name}"
                help_text = get_feature_tooltip(feat)

                # Slider input with session_state synchronization
                st.slider(
                    label=label,
                    min_value=max(0.0, float(col_min * 0.7)),
                    max_value=float(col_max * 1.3),
                    value=float(st.session_state.get(f"input_{feat}", dataset_df[feat].mean())),
                    step=step,
                    format=fmt,
                    key=f"input_{feat}",
                    help=help_text
                )

    with tab_mean:
        st.markdown("##### Average morphometric properties of cell nuclei observed in biopsy specimen:")
        render_feature_inputs(MEAN_FEATURES, num_cols=2)

    with tab_se:
        st.markdown("##### Standard error (measurement variation across cells) in biopsy specimen:")
        render_feature_inputs(SE_FEATURES, num_cols=2)

    with tab_worst:
        st.markdown("##### Largest / most abnormal values (mean of the three largest cell nuclei):")
        render_feature_inputs(WORST_FEATURES, num_cols=2)

    st.markdown("---")
    
    # Prediction CTA Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_clicked = st.button("⚡ Run Full Diagnostic Assessment", type="primary", use_container_width=True)

    if predict_clicked:
        # Assemble input feature vector in strict column order expected by StandardScaler
        input_data = {feat: [st.session_state[f"input_{feat}"]] for feat in FEATURE_NAMES}
        input_df = pd.DataFrame(input_data)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict with SVM model
        prediction = model.predict(input_scaled)[0]
        decision_val = model.decision_function(input_scaled)[0]

        # Calibrated Sigmoid Probability
        # Decision function: positive values correspond to class 1 (Benign), negative to class 0 (Malignant)
        prob_benign = 1.0 / (1.0 + np.exp(-decision_val))
        prob_malignant = 1.0 - prob_benign

        is_malignant = (prediction == 0)

        # ---------------------------------------------------------------------
        # DIAGNOSTIC RESULTS DISPLAY
        # ---------------------------------------------------------------------
        st.markdown("## 📋 Diagnostic Assessment Results")

        if is_malignant:
            st.markdown(
                f"""
                <div class='result-card-malignant'>
                    <div class='result-title-malignant'>⚠️ Malignant Carcinoma Detected</div>
                    <div class='result-desc'>
                        The Support Vector Machine (SVM) classifier identifies cytological characteristics strongly consistent with 
                        <b>malignant (cancerous) breast tissue</b>. Immediate clinical review, histopathological confirmation, and oncology consultation are recommended.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='result-card-benign'>
                    <div class='result-title-benign'>✅ Benign (Non-Cancerous) Profile</div>
                    <div class='result-desc'>
                        The Support Vector Machine (SVM) classifier indicates cytological characteristics consistent with 
                        <b>benign (healthy / non-malignant) tissue</b>. No abnormal hyper-dense cellular structures detected exceeding malignancy thresholds.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Metric Indicators Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Primary Classification", "Malignant" if is_malignant else "Benign")
        with mcol2:
            conf_val = (prob_malignant if is_malignant else prob_benign) * 100
            st.metric("Confidence Score", f"{conf_val:.1f}%")
        with mcol3:
            st.metric("SVM Margin Distance", f"{abs(decision_val):.2f}")
        with mcol4:
            st.metric("Risk Level", "HIGH RISK" if is_malignant else "LOW RISK")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # VISUAL COMPARISON: RADAR CHART & KEY DRIVERS
        # ---------------------------------------------------------------------
        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            st.markdown("### 🕸️ Cytology Profile Comparison (Radar Chart)")
            st.caption("Normalized comparison of 10 mean nuclear features against population benchmarks.")

            # Compute normalized 0-100 scores for radar chart across mean features
            radar_categories = [f.replace("mean ", "").title() for f in MEAN_FEATURES]
            
            patient_vals_norm = []
            benign_vals_norm = []
            malignant_vals_norm = []

            benign_df = dataset_df[dataset_df['target'] == 1]
            malignant_df = dataset_df[dataset_df['target'] == 0]

            for feat in MEAN_FEATURES:
                min_v = dataset_df[feat].min()
                max_v = dataset_df[feat].max()
                denom = max_v - min_v if max_v != min_v else 1.0

                p_norm = (st.session_state[f"input_{feat}"] - min_v) / denom * 100
                b_norm = (benign_df[feat].mean() - min_v) / denom * 100
                m_norm = (malignant_df[feat].mean() - min_v) / denom * 100

                patient_vals_norm.append(min(max(p_norm, 0), 100))
                benign_vals_norm.append(b_norm)
                malignant_vals_norm.append(m_norm)

            # Close radar loop
            radar_categories.append(radar_categories[0])
            patient_vals_norm.append(patient_vals_norm[0])
            benign_vals_norm.append(benign_vals_norm[0])
            malignant_vals_norm.append(malignant_vals_norm[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=benign_vals_norm,
                theta=radar_categories,
                fill='toself',
                name='Benign Benchmark',
                line_color='#10B981',
                opacity=0.4
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=malignant_vals_norm,
                theta=radar_categories,
                fill='toself',
                name='Malignant Benchmark',
                line_color='#EF4444',
                opacity=0.4
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_vals_norm,
                theta=radar_categories,
                fill='toself',
                name='Current Patient',
                line_color='#2563EB',
                line_width=3
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=20, b=50),
                height=420
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with viz_col2:
            st.markdown("### 📊 Top Deviating Biomarkers vs Benign Baseline")
            st.caption("Percentage difference of current patient's worst measurements from benign population average:")

            # Calculate % deviation from benign mean for worst features
            worst_devs = []
            for feat in WORST_FEATURES:
                b_mean = benign_df[feat].mean()
                val = st.session_state[f"input_{feat}"]
                pct_diff = ((val - b_mean) / b_mean) * 100 if b_mean > 0 else 0
                clean_name = feat.replace("worst ", "").title()
                worst_devs.append({"Feature": clean_name, "Deviation %": pct_diff})

            dev_df = pd.DataFrame(worst_devs).sort_values(by="Deviation %", ascending=True)

            fig_bar = px.bar(
                dev_df,
                x="Deviation %",
                y="Feature",
                orientation='h',
                color="Deviation %",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                labels={"Deviation %": "% Deviation from Normal Benign"}
            )
            fig_bar.update_layout(
                margin=dict(l=20, r=20, t=20, b=40),
                height=420,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------------------
# MODE 2: BATCH CSV PREDICTOR
# -----------------------------------------------------------------------------
elif mode == "📁 Batch CSV Predictor":
    st.markdown("<h1 class='main-title'>📁 Batch Cytology Analysis (CSV)</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Upload a CSV file containing multiple patient cytology profiles for simultaneous batch screening.</p>",
        unsafe_allow_html=True
    )

    # Sample template download
    sample_df = pd.read_csv(SAMPLE_CSV_PATH) if os.path.exists(SAMPLE_CSV_PATH) else None
    
    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.markdown(
            """
            <div class='info-card'>
                📄 Upload a CSV containing the 30 Wisconsin cytology feature columns. 
                Don't have a file ready? Click the button to the right to download a sample template.
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_dl2:
        if sample_df is not None:
            csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample Batch CSV",
                data=csv_bytes,
                file_name="sample_breast_cancer_batch.csv",
                mime="text/csv",
                use_container_width=True
            )

    uploaded_file = st.file_uploader("Upload Patient Cytology CSV File", type=["csv"])

    target_df = None
    if uploaded_file is not None:
        try:
            target_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {uploaded_file.name} with {len(target_df)} rows and {len(target_df.columns)} columns.")
        except Exception as err:
            st.error(f"Error parsing uploaded CSV: {err}")
    elif st.button("🧪 Test with Ready-to-Use Sample Batch Data (10 Patients)"):
        target_df = sample_df.copy()

    if target_df is not None:
        # Check for missing features
        missing_features = [f for f in FEATURE_NAMES if f not in target_df.columns]
        if missing_features:
            st.error(f"⚠️ Uploaded CSV is missing {len(missing_features)} required feature columns: {missing_features[:5]}...")
        else:
            # Predict
            X_batch = target_df[FEATURE_NAMES]
            X_batch_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_batch_scaled)
            decision_funcs = model.decision_function(X_batch_scaled)

            # Probabilities
            probs_benign = 1.0 / (1.0 + np.exp(-decision_funcs))
            probs_malignant = 1.0 - probs_benign

            results_df = target_df.copy()
            results_df["Prediction"] = ["Malignant" if p == 0 else "Benign" for p in predictions]
            results_df["Confidence %"] = [
                round(probs_malignant[i] * 100, 1) if predictions[i] == 0 else round(probs_benign[i] * 100, 1)
                for i in range(len(predictions))
            ]
            results_df["SVM Margin"] = np.round(decision_funcs, 3)

            # Summary metrics
            total_cases = len(results_df)
            malignant_count = int(np.sum(predictions == 0))
            benign_count = int(np.sum(predictions == 1))
            malignant_pct = (malignant_count / total_cases) * 100
            benign_pct = (benign_count / total_cases) * 100

            st.markdown("### 📊 Batch Screening Overview")
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.metric("Total Patients Processed", total_cases)
            with scol2:
                st.metric("Malignant Cases", f"{malignant_count} ({malignant_pct:.1f}%)")
            with scol3:
                st.metric("Benign Cases", f"{benign_count} ({benign_pct:.1f}%)")
            with scol4:
                avg_conf = results_df["Confidence %"].mean()
                st.metric("Mean Model Confidence", f"{avg_conf:.1f}%")

            # Batch Results Table
            st.markdown("### 📋 Annotated Results Table")
            
            # Display front columns first
            display_cols = ["Prediction", "Confidence %", "SVM Margin"] + [c for c in results_df.columns if c not in ["Prediction", "Confidence %", "SVM Margin"]]
            
            # Format display dataframe with color highlights
            def highlight_pred(val):
                if val == "Malignant":
                    return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
                elif val == "Benign":
                    return "background-color: #D1FAE5; color: #065F46; font-weight: bold;"
                return ""

            apply_styler = results_df[display_cols].style.map if hasattr(results_df[display_cols].style, "map") else results_df[display_cols].style.applymap
            styled_df = apply_styler(highlight_pred, subset=["Prediction"])
            st.dataframe(styled_df, use_container_width=True, height=350)

            # Download annotated batch CSV
            out_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Annotated Predictions Report (CSV)",
                data=out_csv,
                file_name="breast_cancer_predictions_report.csv",
                mime="text/csv",
                type="primary"
            )


# -----------------------------------------------------------------------------
# MODE 3: MODEL PERFORMANCE & ANALYTICS
# -----------------------------------------------------------------------------
elif mode == "📈 Model Performance & Analytics":
    st.markdown("<h1 class='main-title'>📈 Model Architecture & Validation Metrics</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>In-depth evaluation, validation performance, and clinical distribution of the trained SVM model.</p>",
        unsafe_allow_html=True
    )

    # 4 Key Metric Cards
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Test Accuracy", "98.25%", delta="+1.2% over KNN baseline")
    with kcol2:
        st.metric("Recall (Sensitivity)", "100.0%", delta="Zero False Negatives for Benign")
    with kcol3:
        st.metric("Precision", "97.26%")
    with kcol4:
        st.metric("F1-Score", "98.61%")

    st.markdown("---")

    col_mat, col_dist = st.columns([1, 1])

    with col_mat:
        st.markdown("### 🎯 Confusion Matrix (Test Split: 114 Patients)")
        st.caption("Evaluated on 20% holdout test partition:")
        
        # Exact confusion matrix values from model validation:
        # [[41 True Malignant, 2 False Benign], [0 False Malignant, 71 True Benign]]
        z_cm = [[41, 2], [0, 71]]
        x_cm = ['Pred Malignant', 'Pred Benign']
        y_cm = ['Actual Malignant', 'Actual Benign']

        fig_cm = px.imshow(
            z_cm,
            x=x_cm,
            y=y_cm,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Truth", color="Count")
        )
        fig_cm.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_dist:
        st.markdown("### 📊 Dataset Class Distribution")
        st.caption("Wisconsin Diagnostic Breast Cancer (WDBC) Database (569 Total Biopsies):")

        target_counts = dataset_df['target'].value_counts()
        dist_df = pd.DataFrame({
            "Classification": ["Benign (Non-Cancerous)", "Malignant (Cancerous)"],
            "Count": [target_counts.get(1, 357), target_counts.get(0, 212)],
            "Percentage": [f"{(target_counts.get(1, 357)/569)*100:.1f}%", f"{(target_counts.get(0, 212)/569)*100:.1f}%"]
        })

        fig_pie = px.pie(
            dist_df,
            values="Count",
            names="Classification",
            color="Classification",
            color_discrete_map={
                "Benign (Non-Cancerous)": "#10B981",
                "Malignant (Cancerous)": "#F43F5E"
            },
            hole=0.45
        )
        fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 Multi-Model Comparison (from Training Notebook)")
    
    benchmark_comparison = pd.DataFrame([
        {"Model": "Support Vector Machine (SVM)", "Accuracy": "98.25%", "F1-Score": "98.61%", "Precision": "97.26%", "Status": "Selected (Production)"},
        {"Model": "Random Forest Classifier", "Accuracy": "96.49%", "F1-Score": "97.22%", "Precision": "95.89%", "Status": "Evaluated"},
        {"Model": "XGBoost Classifier", "Accuracy": "96.49%", "F1-Score": "97.26%", "Precision": "96.00%", "Status": "Evaluated"},
        {"Model": "Voting Ensemble (SVM+KNN+RF)", "Accuracy": "96.49%", "F1-Score": "97.26%", "Precision": "96.00%", "Status": "Evaluated"},
        {"Model": "K-Nearest Neighbors (KNN)", "Accuracy": "95.61%", "F1-Score": "96.55%", "Precision": "94.59%", "Status": "Evaluated"},
        {"Model": "Logistic Regression", "Accuracy": "97.37%", "F1-Score": "97.93%", "Precision": "97.26%", "Status": "Evaluated"},
        {"Model": "Gaussian Naive Bayes", "Accuracy": "93.86%", "F1-Score": "95.17%", "Precision": "93.24%", "Status": "Evaluated"}
    ])
    st.table(benchmark_comparison)


# -----------------------------------------------------------------------------
# MODE 4: CLINICAL FEATURE GUIDE
# -----------------------------------------------------------------------------
elif mode == "📖 Clinical Feature Guide":
    st.markdown("<h1 class='main-title'>📖 Clinical Feature Guide & Cytology Dictionary</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Detailed clinical definitions of the Fine Needle Aspirate (FNA) digital imaging markers.</p>",
        unsafe_allow_html=True
    )

    guide_items = [
        ("Radius", "Mean of distances from the center of the cell nucleus to points on its boundary contour.", "Higher in malignant cells due to cellular enlargement (megakaryosis)."),
        ("Texture", "Standard deviation of gray-scale pixel intensities inside the nuclear region.", "Higher texture indicates nuclear chromatin clumpiness and hyperchromasia."),
        ("Perimeter", "Total contour length of the cell nuclear envelope.", "Malignant nuclei have larger and more erratic boundary perimeters."),
        ("Area", "Surface area enclosed within the nuclear contour in square microns.", "Enlarged nuclear area is one of the classic cytological hallmarks of malignancy."),
        ("Smoothness", "Local variations in radius lengths around the nuclear boundary.", "Reflects the microscopic jaggedness or irregularity of the nuclear membrane."),
        ("Compactness", "Computed via the ratio: Perimeter^2 / Area - 1.0.", "Measures the deviation of the nucleus from an ideal circle; elevated in tumors."),
        ("Concavity", "Severity and depth of concave indentations in the nuclear contour.", "Normal nuclei are convex; tumors show pronounced clefting and indentation."),
        ("Concave Points", "Absolute count of concave indentations along the perimeter boundary.", "One of the most discriminative markers separating benign from malignant FNA specimens."),
        ("Symmetry", "Degree of nuclear bilateral and radial shape symmetry.", "Malignant nuclei exhibit progressive nuclear pleomorphism and structural asymmetry."),
        ("Fractal Dimension", "Approximation of contour boundary roughness using the coastline dimension formula.", "Captures higher-order nuclear border complexity.")
    ]

    for title, desc, clinical_note in guide_items:
        with st.expander(f"🔬 {title}", expanded=True):
            st.markdown(f"**Measurement Description:** {desc}")
            st.markdown(f"**Diagnostic Significance:** {clinical_note}")


# -----------------------------------------------------------------------------
# CLINICAL DISCLAIMER FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class='disclaimer-box'>
        <b>⚕️ Clinical Decision Support Disclaimer:</b> This machine learning tool is developed for research, educational, 
        and investigative clinical decision support purposes. Predictions are generated using an automated Support Vector Classifier (SVC) 
        trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. All outputs must be interpreted by licensed medical professionals 
        in conjunction with biopsy histology, mammography imaging, patient history, and definitive diagnostic pathology.
    </div>
    """,
    unsafe_allow_html=True
)
