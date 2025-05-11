import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@st.cache(allow_output_mutation=True)
def load_model():
    model = SimpleNN()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(image)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def main():
    st.title("MNIST Digit Recognition")
    st.write("Upload a 28x28 grayscale image of a digit for prediction.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(img_tensor)
            predicted_digit = torch.argmax(output, dim=1).item()

        st.write(f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    main()
