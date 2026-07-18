import streamlit as st
from PIL import Image
from model import predict

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Alzheimer's MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>

.main{
    background-color:#f4f8fb;
}

.header{
    background:linear-gradient(90deg,#0052D4,#4364F7,#6FB1FC);
    padding:30px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:30px;
}

.prediction-box{
    padding:20px;
    border-radius:12px;
    background:#e8f5e9;
    border-left:8px solid #2e7d32;
    font-size:24px;
    font-weight:bold;
    color:#1b5e20;
}

.info-box{
    background:white;
    padding:20px;
    border-radius:10px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="header">
<h1>🧠 Alzheimer's MRI Classification</h1>
<h4>Deep Learning based Alzheimer's Stage Prediction using MobileNetV2</h4>
</div>
""", unsafe_allow_html=True)

left,right = st.columns([2,1])

with right:

    st.markdown("### 📚 About Alzheimer's")

    st.info("""
Alzheimer's Disease is a progressive neurological disorder that causes memory loss and cognitive decline.

This AI model predicts one of the following stages:

• Non Demented

• Very Mild Dementia

• Mild Dementia

• Moderate Dementia
""")

with left:

    uploaded_file = st.file_uploader(
        "📤 Upload Brain MRI Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )

        if st.button("🔍 Analyze MRI", use_container_width=True):

            with st.spinner("Analyzing MRI..."):

                scores,prediction = predict(image)

            st.markdown(
                f"""
<div class="prediction-box">
Prediction : {prediction}
</div>
""",
                unsafe_allow_html=True
            )

            st.write("")

            st.subheader("📊 Prediction Confidence")

            st.bar_chart(scores)

            st.subheader("📈 Class Probabilities")

            for label,score in scores.items():

                st.progress(float(score))

                st.write(
                    f"**{label}** : {score*100:.2f}%"
                )

st.markdown("---")

st.markdown(
"""
<center>
Made with ❤️ using Streamlit | PyTorch | MobileNetV2
</center>
""",
unsafe_allow_html=True
)
