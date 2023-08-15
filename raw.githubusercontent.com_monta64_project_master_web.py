import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import keras
#Background
st.markdown('<h1 '+ 'style="  color: #191970;  font-family: verdana; font-size: 300%;"'+'>Covid detector from CT scans</h1>', unsafe_allow_html=True)

@st.cache
#classification function
def classification(img, weights_file):
    model = keras.models.load_model(weights_file)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    return prediction
#upload an image to scan
uploaded_file = st.file_uploader("Choose lung CT scan ...", type="jpg")

if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='CT scan.',width=320 )
        st.write("")
        st.write("Classifying...")
        prediction = classification(image, 'C:/users/Monta/Desktop/pfe/model2resnet.h5')
        label = np.argmax(prediction)
        st.write(prediction)
        if label == 0:
            st.sidebar.write("**Covid-19**")
        elif label== 1:
            st.sidebar.write("**Normal**")
        elif label == 2:
        	st.sidebar.write("**Pneumonia**")
#footer
st.sidebar.write('Project by **Montasser Ben Arfia**.')
