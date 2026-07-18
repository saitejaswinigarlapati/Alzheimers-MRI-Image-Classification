import streamlit as st
from PIL import Image
from model import predict

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer's MRI Classification",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("🧠 Alzheimer's MRI Classification")

st.write(
    """
Upload a brain MRI image and the trained MobileNetV2 model
will predict the Alzheimer's stage.
"""
)

st.info("""
Possible Predictions:
- Mild Dementia
- Moderate Dementia
- Non Demented
- Very Mild Dementia
""")

# -------------------------------------------------------
# Upload Image
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    with st.spinner("Predicting..."):

        scores, prediction = predict(image)

    st.success(f"### Prediction: {prediction}")

    st.subheader("Prediction Confidence")

    st.bar_chart(scores)

    st.write("### Class Probabilities")

    for label, score in scores.items():
        st.write(f"**{label}** : {score:.4f}")
