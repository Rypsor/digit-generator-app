# app.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np


# --- WEB APP CONFIGURATION ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND in your script.
st.set_page_config(
    page_title="Digit Generator",
    page_icon="🎨",
    layout="wide"
)


# --- MODEL DEFINITION ---
# IMPORTANT: This must be the EXACT same Generator class definition
# as the one you used for training in your Google Colab notebook.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # These parameters must match your trained model
        self.latent_dim = 100
        self.n_classes = 10
        self.img_shape = (1, 28, 28)

        self.label_embedding = nn.Embedding(self.n_classes, self.n_classes)

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.n_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# --- LOAD THE TRAINED MODEL ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model(model_path):
    device = torch.device('cpu')
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

generator = load_model('generator.pth')
latent_dim = 100

# --- WEB APP INTERFACE ---
st.title("Handwritten Digit Generation using a cGAN")

st.write("""
This web app uses a Conditional Generative Adversarial Network (cGAN) trained on the MNIST dataset to generate images of handwritten digits.
Select a digit from the dropdown menu below and click 'Generate' to see five unique examples created by the AI.
""")

# --- USER INPUT AND IMAGE GENERATION ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Choose a digit to generate", list(range(10)))

if st.sidebar.button("Generate"):
    with st.spinner(f"Generating 5 images of the digit '{selected_digit}'..."):
        num_images = 5
        noise = torch.randn(num_images, latent_dim)
        labels = torch.full((num_images,), selected_digit, dtype=torch.long)
        generated_images = generator(noise, labels)
        generated_images = (generated_images + 1) / 2
        
        st.header(f"Generated Images for Digit: {selected_digit}")
        
        cols = st.columns(num_images)
        for i, image_tensor in enumerate(generated_images):
            img_np = image_tensor.squeeze().detach().numpy()
            with cols[i]:
                st.image(img_np, caption=f"Image {i+1}", width=150)
else:
    st.info("Select a digit and click 'Generate' to start.")