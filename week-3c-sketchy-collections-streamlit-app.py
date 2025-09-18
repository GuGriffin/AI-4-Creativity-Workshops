import os
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import open_clip

# Load pre-trained model and embeddings
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.eval()

# Load embedding and id arrays
met_embeddings = np.load("demo/embeddings/met_embeddings.npy")
met_ids = np.load("demo/embeddings/met_embedding_ids.npy")
id_list = met_ids.tolist()


# After completing all the tasks in Week-3b-CLIP-museum-archive-exercise.ipynb
#  Copy and paste your code for the functions:
# - get_clip_embedding_from_PIL_image()
# - get_id_for_most_similar_item()
# into the space below t
######### Your code goes here:


##########

def get_all_cosine_similarities(embeddings_matrix, embedding_vector):
        dot_product = embeddings_matrix @ embedding_vector
        product_of_magnitudes = np.linalg.norm(embeddings_matrix, axis = 1) * np.linalg.norm(embedding_vector)
        return dot_product / product_of_magnitudes

def main():
    st.title("Sketchy Collections")
    st.write("Inspired by [Polo Sologub's original sketchy collections project](https://ualshowcase.arts.ac.uk/project/316244/cover).")
    st.write("Draw your sketch below to get started:")
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#ffffff",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Process the sketch if available
    if canvas_result.image_data is not None:
        sketch_image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")

        # Generate embedding for the sketch
        if st.button("Find Similar Artwork"):
            with st.spinner("Processing your sketch..."):
                sketch_embedding = get_clip_embedding_from_PIL_image(sketch_image)
                cosine_similarities = get_all_cosine_similarities(met_embeddings, sketch_embedding)
                most_similar_id = get_id_for_most_similar_item(cosine_similarities, id_list)
                
                # Display the result
                if most_similar_id:
                    similar_image_path = os.path.join("class-datasets/met_images", f"{most_similar_id}.jpg")
                    met_url = f"https://www.metmuseum.org/art/collection/search/{most_similar_id}"

                    if os.path.exists(similar_image_path):
                        similar_image = Image.open(similar_image_path)

                        # Display sketch and result side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(sketch_image, caption="Your Sketch", use_container_width=True)
                        with col2:
                            st.image(similar_image, caption=f"Most similar museum item (ID: {most_similar_id})", use_container_width=True)
                            st.markdown(f"[View on MET Website]({met_url})", unsafe_allow_html=True)
                    else:
                        st.error("The similar image file could not be found.")
                else:
                    st.error("No similar artwork found.")

if __name__ == "__main__":
    main()
