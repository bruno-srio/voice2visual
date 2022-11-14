# -*- coding: utf-8 -*-
"""voiceToText.ipynb

# Original file is located at
#    https://colab.research.google.com/drive/1FidpUoFj0rhE4mBMxTJBjM4mCFq8i_Q-

#======================= ⚙️ Config: =======================

# FFmpeg - tool to record, convert and stream audio and video.
pip install ffmpeg

# Installing necessary packages
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/openai/whisper.git 
pip install diffusers==0.2.4
pip install transformers scipy ftfy
pip install "ipywidgets>=7,<8"
pip install tensorflow-gpu

pip install --upgrade tensorflow-gpu

# Trying to debug some errors

pip uninstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip uninstall git+https://github.com/openai/whisper.git 
pip uninstall diffusers==0.2.4
pip uninstall transformers scipy ftfy
pip uninstall "ipywidgets>=7,<8"
pip uninstall tensorflow-gpu
"""

#=======================📥 Importing: =======================
#Importing necessary packages
import streamlit as st
import whisper
import torch
from diffusers import StableDiffusionPipeline
from torch.cuda import is_available
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login


#=======================🎤 Speech to text: =======================

#Authentication of the Stable Diffusion with Hugging Face
notebook_login()

#Check if we are using GPU.
assert is_available(), "You need to enable GPU in Colab or GPU is not available!"

#👨‍💻 Coding:
# Using Whisper to convert audio to text prompt

st.header("🎤 Speech to Image 📷 ")
st.subheader("Turning your imagination into reality!")
st.write("Press the ⏺️ button below to start recording your voice.")
st.write("After you are done, press the button again to stop recording. Wait and check the result! 😃")

# Loading model
model = whisper.load_model('small')

# Loading audio file ()
audio = whisper.load_audio('audioTestPT.m4a') #audioTestEN.m4a for english
#padding audio to 30 seconds
audio = whisper.pad_or_trim(audio)

# Generating spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
st.write(f"Detected language: {max(probs, key=probs.get)}")

# Decoding
options = whisper.DecodingOptions(fp16 = False)
result = whisper.decode(model, mel, options)

# Ready prompt!
prompt = result.text

# Adding tips
prompt += ' hd, 4k resolution, digital drawning'
st.write(prompt) # -> [the text prompt]. hd, 4k resolution, cartoon style

#======================= 🎨 Text to image: =======================

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    revision='fp16',
    torcj_dtype=torch.float16,
    use_auth_token=True
)

# Using pipe we can generate image from text.
pipe = pipe.to("cuda")

with torch.autocast('cuda'):
    image = pipe(prompt)['sample'][0]

# Check the result:
plt.imshow(image)
plt.title(prompt)
plt.axis('off')
plt.savefig('result.jpg')
plt.show()

# Use streamlit to show the image
st.image(image, caption='Generated image', use_column_width=True)
