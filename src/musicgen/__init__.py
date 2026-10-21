import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID="facebook/musicgen-small"
DEVICE="cpu"

PROMPT = "Epic cinematic orchestral soundtrack with powerful string, heavy brass, dramatic percussion, and a heroic atmosphere"

#torch.set_num_thereads(4)

print("CArgando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model= model.to(DEVICE)
model.eval()

print("Preparando el prompt")

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio=model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=200 #4s 750=15s
    )
sample_rate=model.config.audio_encoder.sampling_rate # ~32000 Hz
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
    
