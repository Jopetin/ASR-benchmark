import os
import time
import torch
import whisper
import psutil
import pandas as pd
import soundfile as sf
from jiwer import wer
from tqdm import tqdm
import multiprocessing


# ---- Config ---- #
USER = "Name" # So that we can compare each others 
DATASET_ROOT = "./test-clean"
MODEL_SIZE = "base"   # tiny, base, small, medium, large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = f"temp/benchmark_{USER}_{MODEL_SIZE}_{DEVICE}.csv"
N_REPEATS = 3 # Multiple loops of testing for stability
N_CPU_CORES = multiprocessing.cpu_count()

print(f"Using model '{MODEL_SIZE}' on {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)


# ---- Setup ---- #
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
process = psutil.Process(os.getpid())

# ---- Benchmarking ---- #
for audio_path, ref_text in tqdm(dataset, desc="Benchmarking"):

    
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    run_times, wers, cpu_usages, mem_usages, gpu_mems, gpu_peaks  = [] , [] , [] , [], [], []
    
    for _ in range(N_REPEATS):
        cpu_start = process.cpu_percent(interval=None)
        mem_start = process.memory_info().rss / (1024 * 1024)  # MB
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        start = time.time()
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        infer_time = time.time() - start
        
        cpu_end = process.cpu_percent(interval=None)
        mem_end = process.memory_info().rss / (1024 * 1024)  # MB
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0
        gpu_mems.append(gpu_mem)
        gpu_peaks.append(gpu_peak)

        hyp_text = result["text"].strip()
        clip_wer = wer(ref_text.lower(), hyp_text.lower())
        
        run_times.append(infer_time)
        wers.append(clip_wer)
        cpu_usages.append((cpu_start + cpu_end) / 2)
        mem_usages.append(mem_start - mem_end)
    
    avg_time = sum(run_times) / N_REPEATS
    avg_wer = sum(wers) / N_REPEATS
    avg_cpu = sum(cpu_usages) / N_REPEATS
    avg_mem = sum(mem_usages) / N_REPEATS
    avg_gpu = sum(gpu_mems)/N_REPEATS if DEVICE == "cuda" else 0
    peak_gpu = max(gpu_peaks) if DEVICE == "cuda" else 0

    results.append({
        "file_path": audio_path,
        "audio_sec": duration,
        "infer_sec": avg_time,
        "RTF": avg_time / duration,
        "WER": avg_wer,
        "CPU_%": avg_cpu,
        "Mem_MB": avg_mem,
        "GPU_MB_avg": avg_gpu,
        "GPU_MB_peak": peak_gpu
    })

# ---- Results ---- #
df = pd.DataFrame(results)
avg_wer = df["WER"].mean()
avg_rtf = df["RTF"].mean()
avg_cpu = df["CPU_%"].mean()
avg_mem = df["Mem_MB"].mean()
avg_gpu = df["GPU_MB_avg"].mean()
avg_peak_gpu = df["GPU_MB_peak"].mean()

print("\n=== Benchmark Summary ===")
print(f"Model: {MODEL_SIZE}")
print(f"Device: {DEVICE}")
print(f"Files tested: {len(df)}")
print(f"Average WER: {avg_wer*100:.2f}%")
print(f"Average RTF: {avg_rtf:.2f}")
print(f"Avg CPU usage: {avg_cpu:.2f}% ({avg_cpu/100:.2f} of {N_CPU_CORES} cores)")
print(f"Avg memory increase: {avg_mem:.1f} MB")
if DEVICE == "cuda":
    print(f"Avg GPU Memory: {avg_gpu:.1f} MB (Peak: {avg_peak_gpu:.1f} MB)")

# ---- Save CSV ---- #
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df.to_csv(RESULTS_FILE, index=False)
print(f"\nDetailed results saved to {RESULTS_FILE}")