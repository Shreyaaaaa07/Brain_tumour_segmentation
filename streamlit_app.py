import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import sqlite3
from datetime import datetime
import pandas as pd
import tensorflow.keras.backend as K

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #333;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg,#4F46E5,#7C3AED);
    color: white;
    font-weight: bold;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div style="
background:linear-gradient(90deg,#4F46E5,#7C3AED);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
">

<h1>🧠 Brain Tumor Analysis System</h1>

<h4>
Classification • Segmentation • Tumor Size Analysis
</h4>

</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    classifier_model = tf.keras.models.load_model(
        "models/classifier_model.keras"
    )

    unet_model = tf.keras.models.load_model(
        "models/unet_model.keras",
        compile=False
    )

    return classifier_model, unet_model


classifier_model, unet_model = load_models()



# ==================================================
# PDF REPORT GENERATOR
# ==================================================

def generate_pdf_report(
    tumor_type,
    confidence,
    tumor_area_percent,
    severity
):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Brain Tumor Analysis Report",
             ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Tumor Type: {tumor_type}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, f"Tumor Area: {tumor_area_percent:.2f}%", ln=True)
    pdf.cell(200, 10, f"Severity: {severity}", ln=True)

    pdf_output = pdf.output(dest="S").encode("latin-1")

    return pdf_output
# ==================================================
# GRAD-CAM FUNCTION
# ==================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        pred_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
    )

    heatmap = tf.squeeze(
        heatmap
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= tf.math.reduce_max(
        heatmap
    )

    return heatmap.numpy()

# ==================================================
# DATABASE
# ==================================================

conn = sqlite3.connect(
    "patient_history.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(

id INTEGER PRIMARY KEY AUTOINCREMENT,

patient_name TEXT,

tumor_type TEXT,

confidence REAL,

tumor_area REAL,

severity TEXT,

date TEXT

)
""")

conn.commit()



# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🧠 Brain Tumor AI")

    st.markdown("---")

    st.markdown("""
## 🚀 System Modules

🧠 Tumor Classification

🎯 U-Net Segmentation

📏 Tumor Size Analysis

⚠️ Severity Assessment

📄 Medical Report Generation

🔥 Explainable AI (Grad-CAM)

💾 Database Storage

📋 Patient History

📉 Progress Monitoring

📊 Analytics Dashboard
""")

    st.markdown("---")

    st.success("Final Year Major Project")

# ==================================================
# CLASS LABELS
# ==================================================

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ==================================================
# PATIENT DETAILS
# ==================================================

patient_name = st.text_input(
    "👤 Enter Patient Name"
)

# ==================================================
# FILE UPLOADER
# ==================================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# MAIN PROCESS
# ==================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📤 Uploaded MRI Scan")

    col1, col2 = st.columns([2,1])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        st.info(f"""
        Patient: {patient_name}

        Resolution:
        {image_np.shape[1]} x {image_np.shape[0]}
        """)

    # ==========================================
    # CLASSIFICATION
    # ==========================================

    cls_img = cv2.resize(image_np, (224, 224))
    cls_img = cls_img.astype(np.float32) / 255.0
    cls_img = np.expand_dims(cls_img, axis=0)

    with st.spinner("🔍 Analyzing MRI Scan..."):
        prediction = classifier_model.predict(cls_img)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    tumor_type = classes[class_index]

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Tumor Type": classes,
        "Confidence": prediction[0] * 100
    })

    st.bar_chart(
        prob_df.set_index("Tumor Type")
    )
    st.markdown("---")

    st.subheader("Classification Dashboard")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Tumor Type",
            tumor_type
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with c3:
        st.metric(
            "Status",
            "Normal" if tumor_type=="No Tumor" else "Tumor Found"
        )

    # ==========================================
    # NO TUMOR
    # ==========================================

    if tumor_type == "No Tumor":

        st.success("✅ No Tumor Detected")

    else:

        # ==========================================
        # SEGMENTATION
        # ==========================================

        seg_img = cv2.resize(image_np, (128, 128))
        seg_img = seg_img.astype(np.float32) / 255.0

        if seg_img.shape[-1] == 3:
            seg_img = cv2.cvtColor(
                seg_img,
                cv2.COLOR_RGB2GRAY
            )

        seg_img = np.expand_dims(seg_img, axis=-1)
        seg_img = np.expand_dims(seg_img, axis=0)

        mask = unet_model.predict(seg_img)[0]

        mask = mask.squeeze()

        binary_mask = (
            mask > 0.5
        ).astype(np.uint8)

        # ==========================================
        # TUMOR ANALYSIS
        # ==========================================

        tumor_pixels = np.sum(binary_mask)

        total_pixels = (
            binary_mask.shape[0]
            * binary_mask.shape[1]
        )

        tumor_percentage = (
            tumor_pixels / total_pixels
        ) * 100

        tumor_area_pixels = tumor_pixels

        tumor_area_percent = tumor_percentage

        tumor_diameter = np.sqrt(
            (4 * tumor_area_pixels) / np.pi
        )

        if tumor_area_percent < 5:
            size_category = "Small"
        elif tumor_area_percent < 15:
            size_category = "Medium"
        else:
            size_category = "Large"

        if tumor_area_percent < 5:
            severity = "Low"
        elif tumor_area_percent < 15:
            severity = "Moderate"
        else:
            severity = "High"

        # ==========================================
        # TABS
        # ==========================================

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "🧠 Segmentation",
                "📊 Analysis",
                "📈 Statistics",
                "🔥 Explainable AI",
                "📋 Patient History",
                "📉 Progress Monitoring",
                "📊 Analytics Dashboard"
            ]
        )

        # ==========================================
        # TAB 1
        # ==========================================

        with tab1:

            st.subheader("Segmentation Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Original MRI",
                    use_container_width=True
                )

            with col2:
                st.image(
                    binary_mask * 255,
                    caption="Predicted Tumor Mask",
                    use_container_width=True
                )

                st.markdown("### Tumor Overlay")

                resized_mask = cv2.resize(
                    binary_mask.astype(np.uint8),
                    (image_np.shape[1], image_np.shape[0])
                )

                overlay = image_np.copy()
                overlay[resized_mask == 1] = [255, 0, 0]

                st.image(
                    overlay,
                    caption="Tumor Highlighted in Red",
                    use_container_width=True
                )

        # ==========================================
        # TAB 2
        # ==========================================

        with tab2:

            st.subheader("Tumor Size Analysis")

            colA, colB = st.columns(2)

            with colA:

                st.metric(
                    "Tumor Pixels",
                    int(tumor_area_pixels)
                )

                st.metric(
                    "Tumor Area (%)",
                    f"{tumor_area_percent:.2f}"
                )

            with colB:

                st.metric(
                    "Size Category",
                    size_category
                )

                st.metric(
                    "Approx Diameter",
                    f"{tumor_diameter:.2f} px"
                )

            st.progress(
                min(
                    int(tumor_area_percent),
                    100
                )
            )

            st.write(
                f"Tumor occupies approximately "
                f"{tumor_area_percent:.2f}% "
                f"of the MRI image."
            )

            st.metric(
                "Severity",
                severity
            )

            st.markdown("---")

            st.subheader("🧠 AI Interpretation")

            st.info(
                f"""
                Predicted Tumor Type: {tumor_type}

                Estimated Tumor Area: {tumor_area_percent:.2f}%

                Severity Level: {severity}

                This result is generated by an AI model and should be reviewed by a medical professional.
                """
            )

            if severity == "Low":
                st.success("🟢 Low Severity")

            elif severity == "Moderate":
                st.warning("🟡 Moderate Severity")

            else:
                st.error("🔴 High Severity")

            if severity == "High":
                st.error("⚠️ Immediate consultation with a neurologist is recommended.")

            elif severity == "Moderate":
                st.warning("⚠️ Medical follow-up is advised.")

            else:
                st.success("✅ Low-risk assessment.")
            # SAVE TO DATABASE
            # ==================================================

            if patient_name.strip() != "":

                current_time = datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )

                cursor.execute(
                    """
                    INSERT INTO history
                    (
                        patient_name,
                        tumor_type,
                        confidence,
                        tumor_area,
                        severity,
                        date
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_name,
                        tumor_type,
                        float(confidence),
                        float(tumor_area_percent),
                        severity,
                        current_time
                    )
                )

                conn.commit()

            st.markdown("---")

            st.subheader("Medical Report")

            pdf_data = generate_pdf_report(
                tumor_type,
                confidence,
                tumor_area_percent,
                severity
            )

            st.download_button(
                "📄 Download PDF Report",
                data=pdf_data,
                file_name="Brain_Tumor_Report.pdf",
                mime="application/pdf"
            )

        # ==========================================
        # TAB 3
        # ==========================================

        with tab3:

            st.subheader("Tumor Distribution")

            remaining = 100 - tumor_area_percent

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.pie(
                [tumor_area_percent, remaining],
                labels=["Tumor", "Healthy"],
                autopct="%1.1f%%",
                colors=["red", "green"]
            )

            st.pyplot(fig)

            st.subheader("Tumor Overlay")

            resized_mask = cv2.resize(
                binary_mask.astype(np.uint8),
                (
                    image_np.shape[1],
                    image_np.shape[0]
                )
            )

            overlay = image_np.copy()

            overlay[
                resized_mask == 1
            ] = [255, 0, 0]

            st.image(
                overlay,
                caption="Detected Tumor Region",
                use_container_width=True
            )

        # ==========================================
        # TAB 4 - GRAD CAM
        # ==========================================

        with tab4:

            st.subheader(
                "Grad-CAM Explainable AI"
            )

            heatmap = make_gradcam_heatmap(
                cls_img,
                classifier_model,
                "Conv_1"
            )

            heatmap = cv2.resize(
                heatmap,
                (
                    image_np.shape[1],
                    image_np.shape[0]
                )
            )

            heatmap = np.uint8(
                255 * heatmap
            )

            heatmap = cv2.applyColorMap(
                heatmap,
                cv2.COLORMAP_JET
            )

            superimposed_img = cv2.addWeighted(
                image_np,
                0.6,
                heatmap,
                0.4,
                0
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    image_np,
                    caption="Original MRI",
                    use_container_width=True
                )

            with col2:
                st.image(
                    heatmap,
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )

            with col3:
                st.image(
                    superimposed_img,
                    caption="Overlay Result",
                    use_container_width=True
                )

            st.info(
                f"""
                The model focused on the highlighted regions while predicting
                **{tumor_type}** with **{confidence:.2f}% confidence**.
                """
            )

            st.progress(int(confidence))

            st.write(
                f"Model Confidence: {confidence:.2f}%"
            )

        # ==========================================
        # TAB 5 - PATIENT HISTORY
        # ==========================================

        with tab5:

            st.subheader("Patient History Database")

            df = pd.read_sql_query(
                "SELECT * FROM history ORDER BY id DESC",
                conn
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.metric(
                "Total Records",
                len(df)
            )

            csv = df.to_csv(index=False)

            st.download_button(
                "📥 Download History CSV",
                csv,
                "patient_history.csv",
                "text/csv"
            )

        # ==========================================
        # TAB 6 - PROGRESS MONITORING
        # ==========================================

        with tab6:

            st.subheader("Tumor Progress Monitoring")

            if patient_name.strip() != "":

                patient_df = pd.read_sql_query(
                    f"""
                    SELECT *
                    FROM history
                    WHERE patient_name='{patient_name}'
                    ORDER BY id
                    """,
                    conn
                )

                if len(patient_df) > 1:

                    st.line_chart(
                        patient_df["tumor_area"]
                    )

                    latest = patient_df.iloc[-1]["tumor_area"]
                    previous = patient_df.iloc[-2]["tumor_area"]

                    change = latest - previous

                    st.metric(
                        "Tumor Area Change (%)",
                        f"{latest:.2f}",
                        delta=f"{change:.2f}"
                    )

                    if change > 0:
                        st.error(
                            "Tumor size increased compared to previous scan."
                        )

                    elif change < 0:
                        st.success(
                            "Tumor size reduced compared to previous scan."
                        )

                    else:
                        st.info(
                            "No significant change detected."
                        )

                else:

                    st.warning(
                        "At least 2 records are required for comparison."
                    )

        # ==========================================
        # TAB 7 - ANALYTICS DASHBOARD
        # ==========================================

        with tab7:

            st.subheader("📊 Overall Analytics")

            df = pd.read_sql_query(
                "SELECT * FROM history",
                conn
            )

            if len(df) > 0:

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Patients",
                        len(df)
                    )

                with col2:
                    st.metric(
                        "Average Confidence",
                        f"{df['confidence'].mean():.2f}%"
                    )

                with col3:
                    st.metric(
                        "Average Tumor Area",
                        f"{df['tumor_area'].mean():.2f}%"
                    )

                st.markdown("### Tumor Area Trend")
                st.line_chart(df["tumor_area"])

                st.markdown("### Confidence Trend")
                st.line_chart(df["confidence"])
# ==================================================
# FOOTER
# ==================================================

st.markdown("""
---
### 🧠 Brain Tumor Analysis System

Developed using:

- MobileNetV2 Classification
- U-Net Segmentation
- Grad-CAM Explainability
- Streamlit Web App
- TensorFlow Deep Learning

Final Year Major Project
""")