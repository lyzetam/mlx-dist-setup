import mlx.core as mx
from mlx_lm import load, generate
import socket
import time

def main():
    # Initialize distributed
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()
    
    # Set GPU
    mx.set_default_device(mx.gpu)
    
    if rank == 0:
        print(f"=== MLX Distributed Inference ===")
        print(f"Running on {size} processes")
        print(f"Hosts: {', '.join([f'rank{i}' for i in range(size)])}")
        print("="*40)
    
    # Each rank loads the model
    if rank == 0:
        print("\nLoading model on all ranks...")
    
    start = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start
    
    print(f"[Rank {rank}/{hostname}] Model loaded in {load_time:.2f}s")
    
    # Synchronize after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    # Different prompts for each rank
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning helps us to",
        "The most important technology today is",
        "Distributed computing enables",
        "Apple Silicon chips are",
        "The best programming language is"
    ]
    
    prompt = prompts[rank % len(prompts)]
    
    if rank == 0:
        print(f"\n=== Generating Responses ===")
    
    # Generate response
    start = time.time()
    result = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=50
    )
    gen_time = time.time() - start
    
    # Print results in order
    for i in range(size):
        if rank == i:
            print(f"\n[Rank {rank}/{hostname}]")
            print(f"Prompt: {prompt}")
            print(f"Response: {result}")
            print(f"Generation time: {gen_time:.2f}s")
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync barrier
    
    if rank == 0:
        print("\n=== Inference Complete ===")

if __name__ == "__main__":
    main()
