import numpy as np
import streamlit as st

import librosa
import whisper
import sounddevice as sd

import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Whisper model
whisper_model = whisper.load_model("base")

device = "cpu"

# Put functions from task 2 here
######### Your code goes here:



#########

########## Initialise LLM and write function for getting a response based on a query 

checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
llm_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Function to get LLM response
def get_llm_response(text):
    system_prompt_str = "You are a friendly chatbot that answers questions in single sentences."

    messages = [
        {"role": "system", 
        "content": system_prompt_str,
        },
        {"role": "user", 
        "content": text
        }]

    input_text=tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_token_len = inputs.shape[-1]
    outputs = llm_model.generate(inputs, max_new_tokens=125, temperature=0.5, top_p=0.8, min_p=0.5, do_sample=True)
    out_str = tokenizer.decode(outputs[0][input_token_len:])
    out_str = re.sub(r'(<\|im_start\|>assistant\n)|(<\|im_end\|>)','',out_str)
    return out_str

# -------------------------------------------------
# Application
# -------------------------------------------------

st.title("Voice activated chatbot")

if st.button("Record and Chat"): 
    #
    # Task 2 Step C
    #
    
    st.write("You said:", user_text)
    
    if user_text:
        #
        # Task 2 Step D
        #
