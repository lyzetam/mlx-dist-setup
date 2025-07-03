
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

    if rank == 0:
        print(f"🤖 Llama-3.2-1B-Instruct Custom Prompting")
        print(f"📊 Running on {size} nodes")
        print("=" * 50)

    # Load model
    print(f"[Rank {rank}@{hostname}] Loading Llama 1B model...")
    start_time = time.time()
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    load_time = time.time() - start_time
    print(f"[Rank {rank}@{hostname}] Model loaded in {load_time:.2f}s")

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Use the custom prompt
    prompt = "Write a creative story about a robot learning to paint."

    if rank == 0:
        print(f"\n📝 Prompt: {prompt}")
        print("🎨 Generating creative response...")

    # Generate response with more tokens for creative content
    start_time = time.time()
    response = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=150,  # More tokens for creative content
        repetition_penalty=1.1,
        repetition_context_size=20
    )
    gen_time = time.time() - start_time

    # Calculate metrics
    tokens = len(tokenizer.encode(response))
    speed = tokens / gen_time if gen_time > 0 else 0

    # Display results
    for i in range(size):
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        if rank == i:
            print(f"\n🎭 Response from Node {rank} ({hostname}):")
            print(f"============================================================")
            print(response.strip())
            print(f"============================================================")
            print(f"📊 Stats: {tokens} tokens in {gen_time:.2f}s ({speed:.1f} tok/s)")
            if rank < size - 1:
                print()

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print(f"\n✅ Custom prompting complete!")

if __name__ == "__main__":
    main()
