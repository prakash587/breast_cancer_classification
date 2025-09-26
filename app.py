import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# ------------------------------
# Load Model and Scaler only once
# ------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("breast_cancer_model.keras", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ------------------------------
# Feature Names & Ranges
# ------------------------------
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

feature_ranges = {
    "mean radius": (6.98, 28.11),
    "mean texture": (9.71, 39.28),
    "mean perimeter": (43.79, 188.50),
    "mean area": (143.5, 2501.0),
    "mean smoothness": (0.05, 0.16),
    "mean compactness": (0.02, 0.35),
    "mean concavity": (0.0, 0.43),
    "mean concave points": (0.0, 0.20),
    "mean symmetry": (0.11, 0.30),
    "mean fractal dimension": (0.05, 0.10),
    "radius error": (0.11, 2.87),
    "texture error": (0.36, 4.88),
    "perimeter error": (0.76, 21.98),
    "area error": (6.8, 542.2),
    "smoothness error": (0.0017, 0.031),
    "compactness error": (0.002, 0.135),
    "concavity error": (0.0, 0.40),
    "concave points error": (0.0, 0.05),
    "symmetry error": (0.008, 0.08),
    "fractal dimension error": (0.001, 0.03),
    "worst radius": (7.93, 36.04),
    "worst texture": (12.0, 49.5),
    "worst perimeter": (50.4, 251.2),
    "worst area": (185.2, 4254.0),
    "worst smoothness": (0.07, 0.22),
    "worst compactness": (0.02, 1.06),
    "worst concavity": (0.0, 1.25),
    "worst concave points": (0.0, 0.29),
    "worst symmetry": (0.16, 0.66),
    "worst fractal dimension": (0.06, 0.21),
}

# ------------------------------
# Example Cases
# ------------------------------
malignant_example = [
    17.99, 10.38, 122.80, 1001.0, 0.11840,
    0.27760, 0.3001, 0.14710, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.40, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.60, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.11890
]

benign_example = [
    11.76, 21.60, 74.72, 427.9, 0.08637,
    0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
    0.4062, 1.21, 2.635, 28.47, 0.005857,
    0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
    12.98, 25.72, 82.98, 516.5, 0.1085,
    0.08615, 0.05523, 0.03715, 0.2433, 0.06563
]

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🔬 Breast Cancer Classification App")

# Sidebar for Examples
st.sidebar.header("Examples")
if st.sidebar.button("⚠️ Malignant Example"):
    st.session_state.inputs = malignant_example
if st.sidebar.button("🟢 Benign Example"):
    st.session_state.inputs = benign_example
if st.sidebar.button("🔄 Reset to Min Values"):
    st.session_state.inputs = [feature_ranges[f][0] for f in features]

# Initialize session state
if "inputs" not in st.session_state:
    st.session_state.inputs = [feature_ranges[f][0] for f in features]

# Input form
st.subheader("Enter Tumor Measurements")
cols = st.columns(3)
input_data = []
for i, feature in enumerate(features):
    min_val, max_val = feature_ranges[feature]
    value = cols[i % 3].number_input(
        f"{feature} [{min_val} - {max_val}]",
        min_value=min_val,
        max_value=max_val,
        step=0.01,
        value=float(st.session_state.inputs[i])
    )
    input_data.append(value)
    st.session_state.inputs[i] = value

# Prediction
if st.button("🔍 Predict Tumor Type"):
    input_array = np.asarray(input_data).reshape(1, -1)
    input_std = scaler.transform(input_array)
    prediction = model.predict(input_std)
    predicted_label = np.argmax(prediction)

    st.subheader("🧾 Prediction Result")
    if predicted_label == 0:
        st.error("⚠️ The tumor is **Malignant**")
    else:
        st.success("🟢 The tumor is **Benign**")
