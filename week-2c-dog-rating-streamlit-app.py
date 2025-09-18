import streamlit as st
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# Path to model
# You can change this to your own model that you trained
model_path = 'demo/ckpt/dog_rating_network.pt'

# Code solution for Week 2b Task 2
# You can replace this with your the code for your own dog rating network
class DogRatingNetwork(nn.Module):
    def __init__(self):
        super(DogRatingNetwork, self).__init__()
        self.hidden = nn.Linear(100, 10, bias=True)
        self.output = nn.Linear(10, 1, bias=True) 

    def forward(self, x):
        x = self.hidden(x) 
        x = F.relu(x)
        x = self.output(x) 
        return x

#Image transforms from week 2b  
transform = transforms.Compose(
    [   
        transforms.Resize(10, antialias=True),
        transforms.CenterCrop(10),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
        torch.flatten
    ])

# Create and load model
model = DogRatingNetwork()
model.load_state_dict(torch.load(model_path))

# Title and description
st.title("AI Powered Dog Ratings")
st.write("Upload an image to get a rating of your doggo using an AI model trained on the WeRateDogs dataset")

# Image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform inference
    with torch.no_grad():
        torch_im = transform(image)
        torch_im = torch_im.unsqueeze(0)
        pred = model(torch_im)
        st.write(f'We give this doggo a rating of {pred.item():.2f}/10')
