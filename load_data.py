import pandas as pd
import multiprocessing


N_CPU_CORES = multiprocessing.cpu_count()

data = [("temp/benchmark_Mert_whisper-base_cuda.csv","cuda"),("temp/benchmark_Mert_whisper-base_cpu.csv","cpu"),("temp/benchmark_Mert_wav2vec2_cuda.csv","cuda"),("temp/benchmark_Mert_wav2vec2_cpu.csv","cpu"),("temp/benchmark_Mert_vosk-small_cpu.csv","cpu"),("temp/benchmark_Mert_vosk-medium_cpu.csv","cpu")]

for MODEL_Name, DEVICE in data:
    df = pd.read_csv(MODEL_Name)
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