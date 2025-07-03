import mlx.core as mx
from mlx_lm import load, generate
import socket
import time

def main():
    # Initialize distributed - this should be the first MLX call
    try:
        world = mx.distributed.init()
        rank = world.rank()
        size = world.size()
        hostname = socket.gethostname()
        
        # Set GPU device
        mx.set_default_device(mx.gpu)
        
        print(f"[Rank {rank}/{size}] Starting on {hostname}")
        print(f"[Rank {rank}] GPU: {mx.metal.is_available()}")
        
        # Sync all processes
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))
        
        if rank == 0:
            print(f"\n=== MLX Distributed Inference ===")
            print(f"Running on {size} processes")
        
        # Load model (each process loads its own copy)
        print(f"[Rank {rank}] Loading model...")
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        print(f"[Rank {rank}] Model loaded")
        
        # Sync after model loading
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))
        
        # Different prompts for each rank
        prompts = [
            "What is distributed computing?",
            "How does MLX work on Apple Silicon?",
            "Explain neural networks briefly:",
            "What are the benefits of local AI?",
            "How do GPUs accelerate ML?",
            "What is quantization in ML?"
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
        
        # Print results in order by rank
        for i in range(size):
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync
            if rank == i:
                print(f"\n[Rank {rank}/{hostname}]")
                print(f"Prompt: {prompt}")
                print(f"Response: {result}")
                print(f"Time: {gen_time:.2f}s")
        
        # Final sync
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))
        
        if rank == 0:
            print(f"\n✅ Inference complete on {size} processes!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
