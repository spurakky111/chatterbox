import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# use your GPU (MPS)
model = ChatterboxTTS.from_pretrained(device="mps")

text = "This is a test of the original Chatterbox on my Mac."
wav = model.generate(text)

ta.save("base_test.wav", wav, model.sr)
print("Saved base_test.wav")
