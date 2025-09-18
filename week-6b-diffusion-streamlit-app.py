import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

from stable_diffusion_pytorch import pipeline
from stable_diffusion_pytorch import model_loader

#Generation parameters
device = 'cpu' #@param {"cpu", "cuda", "mps"]
strength = 0.8  #@param { type:"slider", min: 0, max: 1, step: 0.01 }
do_cfg = True  #@param { type: "boolean" }
height = 512  #@param { type: "integer" }
width = 512  #@param { type: "integer" }
sampler = "k_lms"  #@param ["k_lms", "k_euler", "k_euler_ancestral"]
use_seed = False  #@param { type: "boolean" }
if use_seed:
    seed = 42  #@param { type: "integer" }
else:
    seed = None
    

model = model_loader.preload_models(device)

def generate_image(prompt, num_steps, guidance_scale, width, height):
    with torch.no_grad():
        image = pipeline.generate(prompts=prompt,
                  input_images=[], strength=strength,
                  do_cfg=do_cfg, cfg_scale=guidance_scale,
                  height=height, width=width, sampler=sampler,
                  n_inference_steps=num_steps, seed=seed,
                  models=model, device=device, idle_device='cpu')[0]
    return image

# Streamlit UI
st.title("Stable Diffusion Image Generator")

# User inputs
prompt = st.text_input("Enter a prompt:", "A futuristic city at sunset")
num_steps = st.slider("Number of inference steps", min_value=10, max_value=100, value=50)
guidance_scale = st.slider("Guidance scale", min_value=1.0, max_value=14.0, value=7.5)

if st.button("Generate Image"):
    st.write("Generating... This may take a while.")
    image = generate_image([prompt], num_steps, guidance_scale, width, height)
    st.image(image, caption="Generated Image", use_container_width=True)
    
    # Option to save the image
    img_path = "generated_image.png"
    image.save(img_path)
    st.download_button("Download Image", img_path, "generated_image.png")
