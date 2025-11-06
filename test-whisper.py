import whisper
import torch

model = whisper.load_model("base")

result = model.transcribe("./test-clean/121/121726/121-121726-0005.flac")

print(result["text"])