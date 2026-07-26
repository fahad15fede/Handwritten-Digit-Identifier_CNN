import io
import os

import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide"
)

with st.sidebar:
    st.title("Project")

    st.info(
        """
        ### Technologies

        ✅ Streamlit

        ✅ FastAPI

        ✅ Machine Learning
        """
    )

    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=3)
        if response.status_code == 200:
            st.success("Backend Connected")
        else:
            st.error("Backend Unreachable")
    except requests.RequestException:
        st.error("Backend Disconnected")

st.title("✍️ Handwritten Digit Recognition")
st.write("Recognize handwritten digits using Machine Learning.")

st.divider()

option = st.radio(
    "Choose Input Method",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

image = None

if option == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a handwritten digit",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded)

else:
    camera = st.camera_input("Capture Image")

    if camera:
        image = Image.open(camera)

if image:
    st.subheader("Preview")
    st.image(image, width=300)

if image and st.button("🔍 Predict Digit"):
    with st.spinner("Predicting..."):
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            response = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": ("digit.png", buffer, "image/png")},
                timeout=30,
            )

            if response.status_code != 200:
                st.error(f"Prediction failed: {response.text}")
            else:
                result = response.json()
                st.success("Prediction Complete")

                if result.get("phone_number"):
                    st.metric("Detected Number", result["phone_number"])

                digits = result.get("digits", [])
                confidences = result.get("confidence", [])

                if digits:
                    cols = st.columns(len(digits))
                    for index, (digit, confidence) in enumerate(zip(digits, confidences)):
                        with cols[index]:
                            st.metric(f"Digit {index + 1}", digit, f"{confidence}%")

                probabilities = result.get("probabilities", [])
                if probabilities:
                    with st.expander("View probability distribution"):
                        for index, (digit, probs) in enumerate(zip(digits, probabilities)):
                            st.write(f"**Digit {index + 1} ({digit})**")
                            st.bar_chart(
                                {str(i): p for i, p in enumerate(probs)},
                                use_container_width=True,
                            )

        except requests.RequestException as error:
            st.error(f"Could not reach backend at {BACKEND_URL}. Start it with: uvicorn app:app --reload")
            st.caption(str(error))
