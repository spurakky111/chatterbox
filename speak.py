import os, subprocess
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="mps")  # keep GPU

def speak(text: str, out_path: str = "speak_out.wav"):
    wav = model.generate(text)
    ta.save(out_path, wav, model.sr)
    # auto-play on macOS
    subprocess.run(["afplay", out_path])

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'q' to quit): ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue
        print("Generating and playing…")
        speak(text)
