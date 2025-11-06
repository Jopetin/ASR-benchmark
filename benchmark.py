import os
import time
import torch
import whisper
import soundfile as sf
from jiwer import wer
from tqdm import tqdm

DATASET_ROOT = "./test-clean"
MODEL_SIZE = "base"   # tiny, base, small, medium, large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = f"temp/benchmark_{MODEL_SIZE}_{DEVICE}.csv"

print(f"Using model '{MODEL_SIZE}' on {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)

def load_librispeech_dataset(root):
    """Return list of (audio_path, reference_text) pairs."""
    pairs = []
    for dirpath, _, files in os.walk(root):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(dirpath, file)
                # Parse transcript file
                with open(transcript_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            base_id, text = parts
                            flac_file = os.path.join(dirpath, base_id + ".flac")
                            if os.path.exists(flac_file):
                                pairs.append((flac_file, text))
    return pairs

dataset = load_librispeech_dataset(DATASET_ROOT)
print(f"Found {len(dataset)} audio files in {DATASET_ROOT}")

results = []
total_audio_sec = 0.0
total_infer_sec = 0.0
total_wer = 0.0

for audio_path, ref_text in tqdm(dataset, desc="Benchmarking"):
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr

    start = time.time()
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    infer_time = time.time() - start

    hyp_text = result["text"].strip()
    clip_wer = wer(ref_text.lower(), hyp_text.lower())

    results.append((audio_path, duration, infer_time, clip_wer))
    total_audio_sec += duration
    total_infer_sec += infer_time
    total_wer += clip_wer


avg_wer = total_wer / len(results)

print("\n=== Benchmark Summary ===")
print(f"Model: {MODEL_SIZE}")
print(f"Device: {DEVICE}")
print(f"Files tested: {len(results)}")
print(f"Average WER: {avg_wer*100:.2f}%")
print(f"Total audio duration: {total_audio_sec/60:.1f} min")
print(f"Total inference time: {total_infer_sec/60:.1f} min")

# ---- Save CSV ----
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("file_path,audio_sec,infer_sec,wer\n")
    for path, dur, inf, w in results:
        f.write(f"{path},{dur:.2f},{inf:.2f},{w:.4f}\n")

print(f"\nDetailed results saved to {RESULTS_FILE}")