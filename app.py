import streamlit as st
from PIL import Image
from pipeline import classify_image, explain_image, generate_report

st.title("DFU Classification & Report Demo")

uploaded_file = st.file_uploader("Upload a foot image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run pipeline
    prediction, prob = classify_image(image)
    explanation_image = explain_image(image)
    llm_report = generate_report(prediction, prob)

    # Display
    st.subheader("🔎 Classification Result")
    st.write(f"{prediction} (confidence: {prob:.2f})")

    st.subheader("📊 Model Explanation")
    st.image(explanation_image, caption="Explanation Overlay")

    st.subheader("📝 AI Report")
    st.write(llm_report)
