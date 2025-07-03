
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
        print(f"🚀 MLX Distributed Inference Across {size} Nodes")
        print(f"📊 Cluster: {size} processes")
        print("=" * 50)

    # Load model on all nodes
    print(f"[Rank {rank}@{hostname}] Loading model...")
    start_time = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start_time
    print(f"[Rank {rank}@{hostname}] Model loaded in {load_time:.2f}s")

    # Synchronize after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Different prompts for each node
    prompts = [
        "Write a haiku about distributed computing:",
        "Explain the advantages of Apple Silicon for AI:",
        "What makes MLX special for machine learning?",
        "Describe the future of distributed AI:",
        "How does GPU acceleration improve inference?",
        "What are the benefits of multi-node computing?"
    ]

    prompt = prompts[rank % len(prompts)]

    if rank == 0:
        print(f"
🎭 Generating responses across all nodes...")

    # Generate response
    start_time = time.time()
    response = generate(model, tokenizer, prompt, max_tokens=80)
    gen_time = time.time() - start_time

    # Calculate performance metrics
    tokens = len(tokenizer.encode(response))
    speed = tokens / gen_time if gen_time > 0 else 0

    # Display results in rank order
    for i in range(size):
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync

        if rank == i:
            print(f"
🖥️  Node {rank} ({hostname}):")
            print(f"❓ Prompt: {prompt}")
            print(f"🤖 Response: {response.strip()}")
            print(f"⚡ Performance: {speed:.1f} tokens/sec ({gen_time:.2f}s)")
            print("-" * 50)

    # Final sync
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print(f"
✅ Distributed inference complete!")
        print(f"🎉 Generated {size} responses across Mac cluster")

if __name__ == "__main__":
    main()
