import whisper

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
