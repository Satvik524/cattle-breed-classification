import urllib.request
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os


st.set_page_config(
    page_title ="Indian Cattle Breed Classifier",
    page_icon = "🐄",
    layout = "centered"
)

@st.cache_resource
# def load_model():
#     model_path = "/Users/satviksingh/Desktop/cattle_breed/convnext_model_improved.keras"
#     return tf.keras.models.load_model(model_path,compile = False)
# compile = False prevents throwing error if the coustom loss fucntion is not defind here
   

def load_cloud_model():
    model_path = "cattle_breed_model.keras"
    model_url = 'https://huggingface.co/Shogun007/Indian_Cattle_Breed_classifier/resolve/main/convnext_model_improved.keras?download=true'

    if not os.path.exists(model_path):
        with st.spinner("Downloading model weights from Hugging Face... (This only happens once)"):
           urllib.request.urlretrieve(model_url, model_path)
    
    return tf.keras.models.load_model(model_path,compile = False)


with st.spinner("Loading the ConvNext model please wait...."):
    model = load_cloud_model()


# very important to define the class names in the same order as they were used during model training. i.e after spliting the dataset into train and test sets, the class names were sorted in alphabetical order. So, we need to maintain the same order here as well.
Class_names = ['Amritmahal', 'Ayrshire', 'Bargur', 'Dangi', 'Deoni', 'Gir', 'Hallikar', 'Hariana', 'Himachali Pahari', 'Kangayam', 'Kankrej', 'Kenkatha', 'Khariar', 'Khillari', 'Konkan Kapila', 'Kosali', 'Krishna_Valley', 'Ladakhi', 'Lakhimi', 'Malnad_gidda', 'Mewati', 'Nari', 'Nimari', 'Ongole', 'Poda Thirupu', 'Pulikulam', 'Punganur', 'Purnea', 'Rathi', 'Red kandhari', 'Red_Sindhi', 'Sahiwal', 'Shweta Kapila', 'Tharparkar', 'Umblachery', 'Vechur', 'bachaur', 'badri', 'bhelai', 'dagri', 'gangatari', 'gaolao', 'ghumsari', 'kherigarh', 'malvi', 'motu', 'nagori', 'ponwar', 'siri', 'thutho']
def preprocessing_image(image):

    target_size = (224,224)
    if image.mode!= 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = np.expand_dims(image_arr,axis=0) # Reshapes to (1, 224, 224, 3)

    return image_arr

# Web UI layout
st.title("🐄 Indian Cattle Breed Identification")
st.write("Upload a clear photograph of an Indian cattle breed, and the model will analyze its characteristics.")

uploaded_file = st.file_uploader("choose the image...",type=['jpg','jpeg','png'])

if uploaded_file is not None:
    # displaying the uploaded image
    image = Image.open(uploaded_file)
    st.image(image,caption='Uploaded file',use_column_width=True)

    st.write('')

    if st.button('Identify the Breed'):

        processed_image = preprocessing_image(image)
        prediction = model.predict(processed_image)

        probabilities = prediction[0] # as the prediction is softmax
        best_index = np.argmax(probabilities)
        predicted_breed = Class_names[best_index]
        confidence = 100 * probabilities[best_index]

        st.success(f"**Predicted Breed:** {predicted_breed}")
        st.info(f"**Confidence Level:** {confidence:.2f}%")
            
            # Breakdown of Top 3 classification possibilities
        st.write("### Top Possibilities:")
        top_k_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_k_indices:
            breed_name = Class_names[idx]
            conf_score = float(probabilities[idx])

            col1,col2 = st.columns([1,3])
            with col1:
                st.write(f"**{breed_name}**")
                st.caption(f"{conf_score * 100:.1f}%")
            with col2:
                st.progress(conf_score)
    
    












