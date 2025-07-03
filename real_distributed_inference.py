
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
        print(f"🚀 MLX Distributed Inference with 1B Model")
        print(f"Processes: {size} across cluster")
        print("=" * 50)

    # Load the 1B model on all processes
    if rank == 0:
        print("📦 Loading Llama-3.2-1B model on all nodes...")

    start_time = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start_time

    print(f"[Rank {rank}/{hostname}] Model loaded in {load_time:.2f}s")

    # Synchronize after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Different interesting prompts for each rank
    prompts = [
        "Write a short poem about artificial intelligence:",
        "Explain quantum computing in simple terms:",
        "What are the benefits of distributed computing?",
        "How does machine learning work?",
        "Describe the future of technology:",
        "What makes Apple Silicon special for AI?"
    ]

    prompt = prompts[rank % len(prompts)]

    if rank == 0:
        print(f"\n🎭 Generating responses to different prompts...")

    # Generate response
    start_time = time.time()
    response = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=100
    )
    gen_time = time.time() - start_time

    # Calculate tokens per second
    response_tokens = len(tokenizer.encode(response))
    tokens_per_sec = response_tokens / gen_time if gen_time > 0 else 0

    # Display results in rank order
    for i in range(size):
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync barrier

        if rank == i:
            print(f"\n🤖 [Rank {rank} on {hostname}]")
            print(f"📝 Prompt: {prompt}")
            print(f"💬 Response: {response}")
            print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec ({gen_time:.2f}s)")
            print("-" * 50)

    # Final sync and summary
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print(f"\n✅ Distributed inference complete!")
        print(f"🎉 Successfully generated {size} different responses")

if __name__ == "__main__":
    main()
