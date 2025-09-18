# This app is adapted using the following codebase: https://github.com/AIDevBytes/LLava-Image-Analyzer

import streamlit as st
from util.image_helper import create_temp_file
from util.llm_helper import analyze_image_file, stream_parser

page_title = 'LLAVA Image analyser'

# configures page settings
st.set_page_config(
    page_title=page_title,
    initial_sidebar_state="expanded",
)

# page title
st.title(page_title)

st.markdown("#### Select an image file to analyze.")

# displays file upload widget
uploaded_file = st.file_uploader("Choose image file", type=['png', 'jpg', 'jpeg'] )

image_model = 'llava:7b'

if chat_input := st.chat_input("What would you like to ask?"):
    if uploaded_file is None:
        st.error('You must select an image file to analyze!')
        st.stop()

    # Color formatting example https://docs.streamlit.io/library/api-reference/text/st.markdown
    with st.status(":red[Processing image file.]", expanded=True) as status:
        st.write(":orange[Analyzing Image File...]")

        # TASK 1: ---------------
        stream = analyze_image_file(uploaded_file, model=image_model, user_prompt=chat_input)
        # ------------------------

        # Task 2: ----------------
        stream_output = st.write_stream(stream_parser(stream))
        # ------------------------
