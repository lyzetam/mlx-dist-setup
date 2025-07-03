
import mlx.core as mx
from mlx_lm import load, generate
import socket
import time

def main():
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()

    mx.set_default_device(mx.gpu)

    # Load model and measure time
    load_start = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - load_start

    # Get GPU memory after loading
    try:
        mem_info = mx.metal.get_memory_info()
        gpu_memory = mem_info["allocated"] / 1024 / 1024  # MB
    except:
        gpu_memory = 0

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Performance test - generate multiple responses
    prompt = f"Explain distributed computing for node {rank}:"

    total_tokens = 0
    total_time = 0
    runs = 3

    for i in range(runs):
        start = time.time()
        response = generate(model, tokenizer, prompt, max_tokens=50)
        gen_time = time.time() - start

        tokens = len(tokenizer.encode(response))
        total_tokens += tokens
        total_time += gen_time

        mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync

    avg_speed = total_tokens / total_time if total_time > 0 else 0

    print(f"PERF|RANK_{rank}|HOST_{hostname}|LOAD_{load_time:.2f}s|GPU_{gpu_memory:.1f}MB|SPEED_{avg_speed:.1f}tok/s|RUNS_{runs}")

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

if __name__ == "__main__":
    main()
