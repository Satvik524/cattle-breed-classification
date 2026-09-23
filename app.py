import requests
import streamlit as st
from PIL import Image


API_URL = "http://api:8000/predict" 


st.set_page_config(
    page_title="Indian Cattle Breed Classifier",
    page_icon="🐄",
    layout="centered"
)


st.title("🐄 Indian Cattle Breed Identification")
st.write(
    "Upload a clear photograph of an Indian cattle breed and the model will analyze its characteristics."
)


uploaded_file = st.file_uploader(
    "Choose the image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded file",
        width="stretch"
    )

    if st.button("Identify the Breed"):

        files = {
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        try:
            response = requests.post(
                API_URL,
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"**Predicted Breed:** "
                    f"{result['predicted_breed']}"
                )

                st.info(
                    f"**Confidence Level:** "
                    f"{result['confidence'] * 100:.2f}%"
                )

                st.write("### Top Possibilities:")

                for prediction in result["top_predictions"]:

                    breed = prediction["breed"]
                    confidence = prediction["confidence"]

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.write(f"**{breed}**")
                        st.caption(
                            f"{confidence * 100:.1f}%"
                        )

                    with col2:
                        st.progress(confidence)

            else:

                error = response.json()

                st.error(
                    error.get(
                        "detail",
                        "Prediction failed."
                    )
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the API is running."
            )