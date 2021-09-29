import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        st.write(prediction)
        
        return prediction

model = tf.keras.models.load_model('C:/Users/admin/OneDrive/Desktop/covid 19 detector/Covid-19-app/Covid-app/my_model.hdf5')

st.write("""
         # Covid_19 Detector
         """
         )

st.write("This is a simple image classification web app to predict Covid_19 ")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Normal!")
    else:
        st.write("It is a Covid_19!")
    
st.text("Probability (0: Normal, 1: Covid_19)")
#st.write(prediction)



