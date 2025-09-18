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

def record_audio(duration=5, samplerate=44100):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    st.write("Recording...")
    sd.wait()
    st.write('Recording complete.')
    return audio, samplerate

def preprocess_audio(audio_input, original_sr=44100, target_sr=16000):
    audio_input = audio_input.squeeze()
    audio_input = audio_input.astype(np.float32) / 32768.0
    audio_input = librosa.resample(audio_input, orig_sr=original_sr, target_sr=target_sr)
    audio_input = whisper.pad_or_trim(audio_input)
    return audio_input

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
    audio, samplerate = record_audio()
    audio = preprocess_audio(audio)
    result = whisper_model.transcribe(audio)
    user_text = result['text']
    
    st.write("You said:", user_text)
    
    if user_text:
        llm_response = get_llm_response(user_text)
        st.write("Chatbot:", llm_response)
