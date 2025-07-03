
import mlx.core as mx
from mlx_lm import load, generate
import socket
import time
import sys

def main():
    # Initialize distributed
    world = mx.distributed.init()
    rank = world.rank()
    size = world.size()
    hostname = socket.gethostname()

    # Set GPU
    mx.set_default_device(mx.gpu)

    if rank == 0:
        print("🎭 MLX DISTRIBUTED DEMO - 1B MODEL INFERENCE")
        print("=" * 55)
        print(f"📊 Cluster: {size} processes across nodes")
        print(f"🤖 Model: Llama-3.2-1B-Instruct-4bit")
        print("=" * 55)

    # Load model with timing
    if rank == 0:
        print("\n📦 Loading model on all processes...")

    start = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start

    print(f"✅ [Rank {rank}/{hostname}] Loaded in {load_time:.2f}s")

    # Sync after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Creative prompts for demonstration
    creative_prompts = [
        "Write a haiku about machine learning:",
        "Explain why distributed computing is powerful in one sentence:",
        "What\'s the coolest thing about Apple Silicon?",
        "Describe the future of AI in 2030:",
        "How does MLX make AI development easier?",
        "What makes this distributed setup special?"
    ]

    my_prompt = creative_prompts[rank % len(creative_prompts)]

    if rank == 0:
        print(f"\n🎨 Generating creative responses...")

    # Generate with timing
    start = time.time()
    response = generate(
        model, 
        tokenizer, 
        my_prompt, 
        max_tokens=80
    )
    gen_time = time.time() - start

    # Calculate performance metrics
    tokens = len(tokenizer.encode(response))
    speed = tokens / gen_time if gen_time > 0 else 0

    # Display results in synchronized order
    for i in range(size):
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync

        if rank == i:
            print(f"\n🌟 Process {rank} on {hostname}")
            print(f"❓ Prompt: {my_prompt}")
            print(f"🤖 Response: {response.strip()}")
            print(f"⚡ Performance: {speed:.1f} tok/s ({gen_time:.2f}s, {tokens} tokens)")
            print("-" * 50)

    # Final synchronization and celebration
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print(f"\n🎉 SUCCESS! Distributed MLX inference complete!")
        print(f"📈 Generated {size} unique responses across your Mac cluster")
        print("✨ This demonstrates true distributed AI on Apple Silicon!")
        print("=" * 55)

if __name__ == "__main__":
    main()
