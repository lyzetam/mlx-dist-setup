
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
        print("🌐 TRUE DISTRIBUTED MLX INFERENCE")
        print(f"📊 Nodes: {size} ({', '.join(['mbp.local', 'mm1.local', 'mm2.local'])})")
        print("=" * 50)

    # Load model
    print(f"[Rank {rank}@{hostname}] Loading Llama 1B model...")
    start_time = time.time()
    try:
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        load_time = time.time() - start_time
        print(f"[Rank {rank}@{hostname}] ✅ Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"[Rank {rank}@{hostname}] ❌ Model loading failed: {e}")
        return

    # Synchronize after loading
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Different prompt variations for each node
    prompt_variations = [
        "Write a haiku about distributed computing across multiple Macs",
        "Write a haiku about distributed computing across multiple Macs - focus on collaboration",
        "Write a haiku about distributed computing across multiple Macs - emphasize speed and efficiency"
    ]

    prompt = prompt_variations[rank % len(prompt_variations)]

    if rank == 0:
        print("🎭 Generating unique responses on each Mac...")

    # Generate response
    start_time = time.time()
    try:
        response = generate(model, tokenizer, prompt, max_tokens=120)
        gen_time = time.time() - start_time

        tokens = len(tokenizer.encode(response))
        speed = tokens / gen_time if gen_time > 0 else 0

        # Display results from each node in order
        for i in range(size):
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))  # Sync

            if rank == i:
                print(f"\n🖥️  Mac {rank} ({hostname}):")
                print(f"📝 Prompt: {prompt}")
                print(f"🎨 Response: {response.strip()}")
                print(f"⚡ Performance: {speed:.1f} tok/s ({gen_time:.2f}s, {tokens} tokens)")
                print("-" * 60)

        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        if rank == 0:
            print("🎉 TRUE DISTRIBUTED INFERENCE COMPLETE!")
            print(f"✅ {size} Macs generated {size} unique responses simultaneously")

    except Exception as e:
        print(f"[Rank {rank}@{hostname}] ❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
