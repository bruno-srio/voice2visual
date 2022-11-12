#Necessary packages

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"
pip install git+https://github.com/openai/whisper.git 
pip install diffusers==0.2.4
pip install transformers scipy ftfy
pip install "ipywidgets>=7,<8"


#Convert audio to text
import whisper
from huggingface_hub import notebook_login
from torch.cuda import is_available

notebook_login()
assert is_available(), 'GPU is not available.'

# loading model
model = whisper.load_model('small')

# loading audio file ()
audio = whisper.load_audio('/content/drive/MyDrive/Colab Notebooks/audioTest.m4a')
# padding audio to 30 seconds
audio = whisper.pad_or_trim(audio)

# generating spectrogram
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decoding
options = whisper.DecodingOptions(fp16 = False)
result = whisper.decode(model, mel, options)

# ready prompt!
prompt = result.text

# adding tips
prompt += ' hd, 4k resolution, cartoon style'
print(prompt) # -> [the text prompt]. hd, 4k resolution, cartoon style

#Convert text to image

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    revision='fp16',
    torcj_dtype=torch.float16,
    use_auth_token=True
)
#Using pipe we can generate image from text.
pipe = pipe.to("cuda")

with torch.autocast('cuda'):
    image = pipe(prompt)['sample'][0]


#Check the result
import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(prompt)
plt.axis('off')
plt.savefig('result.jpg')
plt.show()
