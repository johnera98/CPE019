import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

st.title("MNIST Digit Recognizer")
st.write("Upload a 28x28 pixel handwritten digit image (black on white).")

# Load trained model
model = tf.keras.models.load_model('mnist_model.h5')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert for MNIST (white background, black digit)
    image = image.resize((28, 28))  # Resize to 28x28

    img_array = np.array(image) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)

    # Display image
    st.image(image, caption="Processed Image", width=150)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    st.write(f"Predicted Digit: **{predicted_digit}**")
