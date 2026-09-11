import os
import torch
import time

size = 5000

# Always works (Runs on CPU RAM)
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

# --- CPU BENCHMARK ---
start_time = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_time = time.perf_counter() - start_time
print(f"CPU Time: {cpu_time:.3f}s")

# --- GPU BECHMARK (Auto-Detected) ---
if torch.cuda.is_available():
  print(f"\n[GPU Detected]: {torch.cuda.get_device_name(0)}\n")

  # Send data to T4 GPU
  a_gpu = a_cpu.to("cuda")
  b_gpu = b_cpu.to("cuda")

  # CUDA Hardware Timers
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  start_event.record()
  c_gpu = a_gpu @ b_gpu
  end_event.record()

  torch.cuda.synchronize()

  gpu_time_ms = start_event.elapsed_time(end_event)
  gpu_time_s = gpu_time_ms / 1000.0

  print(f"GPU Time: {gpu_time_s:.3f}s ({gpu_time_ms:.2f} ms)")
  print(f"Speedup: {cpu_time / gpu_time_s:.1f}x")
else:
  print("\n[GPU Not Detected]")

if not torch.cuda.is_available():
  print("CUDA is not available. Switch runtime to T4 GPU for the full GPU benchmark.")
  try:
    total_ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    total_ram_gb = total_ram_bytes / (1024 ** 3)
    print(f"System RAM: {total_ram_gb:.2f} GB")
  except (AttributeError, ValueError):
    print("System RAM: unavailable on this platform")

  print("\n--- CPU-ONLY INTERPRETATION ---")
  print("This environment is running without CUDA, so GPU VRAM and speedup cannot be measured here.")
  print("For model sizing, the fp16 rule of thumb is still: ~2 bytes per parameter.")
  print("That means a 16 GB GPU can hold roughly 8 billion fp16 parameters before overhead.")
  print("In practice, you need extra space for activations, gradients, optimizer states, and KV cache.")
  print("If you want a real GPU result, run this same script in Google Colab with a T4 GPU.")
else:
  # 1. Hardware VRAM Inspection
  device = 0
  props = torch.cuda.get_device_properties(device)

  total_memory_bytes = props.total_memory
  total_memory_gb = total_memory_bytes / (1024 ** 3)

  # Check current allocation
  allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
  reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
  free_memory_gb = total_memory_gb - reserved_gb

  print(f"--- GPU VRAM STATUS ---")
  print(f"Device Name:            {props.name}")
  print(f"Total VRAM:             {total_memory_gb:.2f} GB")
  print(f"Currently Allocated:    {allocated_gb:.2f} GB")
  print(f"Currently Reserved:     {reserved_gb:.2f} GB")
  print(f"Usable Free Memory:     {free_memory_gb:.2f} GB\n")

  # 2. Estimate Max Model Size
  # (Rule of Thumb: 2 Bytes / Parameter in FP16)
  # Reserve 20% for KV Cache, activations, and PyTorch CUDA context overhead
  usable_vram_for_weights = free_memory_gb * 0.80
  bytes_per_param = 2 # FP16/ BF16

  max_params_billions = (usable_vram_for_weights * (1024 ** 3)) / (bytes_per_param * 1e9)

  print(f"--- MODEL CAPACITY ESTIMATE (FP16 / BF16) ---")
  print(f"Usable VRAM for Weights: ~{usable_vram_for_weights:.2f} GB (leaving 20% for KV cache & overhead)")
  print(f"Estimated Max Model:     ~{max_params_billions:.2f} Billion Parameters\n")

  # Common Model Reference Check
  print("--- WHAT FITS IN THIS VRAM? ---")
  models = [
      ("1.8B Model (e.g., Qwen1.5-1.8B)", 1.8),
      ("3B Model (e.g., Llama-3.2-3B)", 3.0),
      ("7B / 8B Model (e.g., Llama-3-8B / Mistral-7B)", 7.5),
      ("13B / 14B Model (e.g., Llama-2-13B)", 13.0)
  ]

  for name, params in models:
      vram_needed = (params * 1e9 * bytes_per_param) / (1024 ** 3)
      fits = "YES" if vram_needed <= usable_vram_for_weights else "NO (Requires Quantization)"
      print(f"• {name,-45} | Weights: {vram_needed:.2f} GB | Fits: {fits}")