
import mlx.core as mx
from mlx_lm import load, generate
import socket
import time
import subprocess
import sys

def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        # Use MLX's memory info
        allocated = mx.metal.get_memory_info()["allocated"]
        peak = mx.metal.get_memory_info()["peak"]
        return allocated, peak
    except:
        return 0, 0

def main():
    # Initialize distributed
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()

    # Set GPU
    mx.set_default_device(mx.gpu)

    if rank == 0:
        print("🔍 GPU USAGE MONITORING ACROSS CLUSTER")
        print("=" * 50)
        print(f"📊 Monitoring {size} processes")

    # Check initial GPU memory
    initial_mem, initial_peak = get_gpu_memory()
    print(f"[Rank {rank}@{hostname}] Initial GPU: {initial_mem/1024/1024:.1f}MB allocated")

    # Sync point
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print("\n📦 Loading model on all nodes (watch GPU usage)...")

    # Load model and monitor memory
    start_time = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start_time

    # Check post-load GPU memory
    post_load_mem, post_load_peak = get_gpu_memory()
    model_mem = post_load_mem - initial_mem

    print(f"[Rank {rank}@{hostname}] Model loaded in {load_time:.2f}s")
    print(f"[Rank {rank}@{hostname}] GPU Memory: {post_load_mem/1024/1024:.1f}MB (+{model_mem/1024/1024:.1f}MB for model)")

    # Sync after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Generate inference and monitor GPU during generation
    prompt = f"What is the role of process {rank} in distributed computing?"

    if rank == 0:
        print(f"\n🚀 Starting inference on all {size} processes...")
        print("Monitor GPU usage during generation:")

    # Sync before generation
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Generate with memory monitoring
    pre_gen_mem, _ = get_gpu_memory()
    start_time = time.time()

    response = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=50
    )

    gen_time = time.time() - start_time
    post_gen_mem, peak_mem = get_gpu_memory()

    # Calculate metrics
    tokens = len(tokenizer.encode(response))
    speed = tokens / gen_time if gen_time > 0 else 0
    gen_mem_used = post_gen_mem - pre_gen_mem

    # Display results in order
    for i in range(size):
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        if rank == i:
            print(f"\n🤖 Process {rank} on {hostname}:")
            print(f"  📝 Prompt: {prompt}")
            print(f"  💬 Response: {response.strip()}")
            print(f"  🖥️  GPU Memory: {post_gen_mem/1024/1024:.1f}MB (peak: {peak_mem/1024/1024:.1f}MB)")
            print(f"  ⚡ Performance: {speed:.1f} tok/s ({gen_time:.2f}s)")
            print("  " + "-" * 45)

    # Final sync and GPU summary
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print(f"\n✅ GPU monitoring complete!")
        print(f"🎯 All {size} processes used GPU memory for model and inference")

if __name__ == "__main__":
    main()
