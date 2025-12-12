import os
import time
import torch
import psutil
import pandas as pd
import soundfile as sf
from jiwer import wer
from tqdm import tqdm
import multiprocessing
import numpy as np
import tracemalloc

# !Note! To use veck you need to download a model and add the path to it in the model loading section 


# ---- Config ---- #
USER = "Name" # So that we can compare each others 
DATASET_ROOT = "./test-clean"

MODEL_TYPE = "vosk"   # "wav2vec2", "vosk", "whisper" 
if MODEL_TYPE == "whisper":
    MODEL_SIZE = "base"   # tiny, base, small, medium, large
    MODEL_Name = f"{MODEL_TYPE}-{MODEL_SIZE}"
else:
    MODEL_Name = MODEL_TYPE
    
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"
if MODEL_TYPE == "vosk":
    DEVICE = "cpu"
    
RESULTS_FILE = f"temp/benchmark_{USER}_{MODEL_Name}_{DEVICE}.csv"
N_REPEATS = 1 # Multiple loops of testing for stability
N_CPU_CORES = multiprocessing.cpu_count()

print(f"Using model '{MODEL_Name}' on {DEVICE}...")



# ---- Model Loading ---- #
if MODEL_TYPE == "whisper":
    import whisper
    
    load_start = time.time()
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    load_time = time.time() - load_start

elif MODEL_TYPE == "wav2vec2":
    import torchaudio
    
    load_start = time.time()
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(DEVICE)
    labels = bundle.get_labels()
    load_time = time.time() - load_start

elif MODEL_TYPE == "vosk":
    import vosk
    import json
    
    load_start = time.time()
    model = vosk.Model("vosk/vosk-model-en-us-0.22-lgraph")
    load_time = time.time() - load_start

else:
    raise ValueError("MODEL_TYPE must be 'whisper', 'wav2vec2', or 'vosk'")
print(f"{MODEL_Name} load time: {load_time:.2f} sec")



# ---- Transcription Function ---- #
def transcribe_audio(audio_path):
    """Unified transcription function."""
    # ---- Whisper ---- #
    if MODEL_TYPE == "whisper":
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        return result["text"]

    # ---- wav2vec2 ---- #
    elif MODEL_TYPE == "wav2vec2":
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(DEVICE)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        with torch.inference_mode():
            emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
        tokens = torch.argmax(emissions[0], dim=-1)
        prev_token = None
        decoded = []
        for t in tokens:
            token = t.item()
            if token != prev_token and token != 0:  
                decoded.append(labels[token])
            prev_token = token
        text = "".join(decoded).replace("|", " ").strip()
        return text

    # ---- vosk ---- #
    elif MODEL_TYPE == "vosk":    
        wf, sr = sf.read(audio_path)
        wf_int16 = (wf * 32768).astype("int16")
        rec = vosk.KaldiRecognizer(model, sr)
        rec.AcceptWaveform(wf_int16.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()
    
    
    
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

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        tracemalloc.start()
        start = time.time()
        
        hyp_text = transcribe_audio(audio_path)
        infer_time = time.time() - start
        
        
        cpu_end = process.cpu_percent(interval=None)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0
        gpu_mems.append(gpu_mem)
        gpu_peaks.append(gpu_peak)

        clip_wer = wer(ref_text.lower(), hyp_text.lower())
        
        run_times.append(infer_time)
        wers.append(clip_wer)
        cpu_usages.append((cpu_start + cpu_end) / 2)
        mem_usages.append(peak / (1024 * 1024))
    
    avg_time = np.mean(run_times)
    avg_wer = np.mean(wers)
    avg_cpu = np.mean(cpu_usages)
    avg_mem = np.mean(mem_usages)
    avg_gpu = np.mean(gpu_mems) if DEVICE == "cuda" else 0
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
print(f"Model: {MODEL_Name}")
print(f"Device: {DEVICE}")
print(f"Files tested: {len(df)}")
print(f"Average WER: {avg_wer*100:.2f}%")
print(f"Average RTF: {avg_rtf:.2f}")
print(f"Avg CPU usage: {avg_cpu:.2f}% ({avg_cpu/100:.2f} of {N_CPU_CORES} cores)")
print(f"Avg memory usage: {avg_mem:.2f} MB")
if DEVICE == "cuda":
    print(f"Avg GPU Memory: {avg_gpu:.1f} MB (Peak: {avg_peak_gpu:.1f} MB)")



# ---- Save CSV ---- #
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
df.to_csv(RESULTS_FILE, index=False)
print(f"\nDetailed results saved to {RESULTS_FILE}")