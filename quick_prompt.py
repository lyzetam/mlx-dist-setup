
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
        print("🤖 Llama 1B Custom Prompting Started")
        print(f"📊 Nodes: {{size}}")
        print("=" * 40)

    # Load model
    start_time = time.time()
    try:
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        load_time = time.time() - start_time

        if rank == 0:
            print(f"✅ Model loaded in {{load_time:.1f}}s")
    except Exception as e:
        print(f"❌ Model loading failed on rank {{rank}}: {{e}}")
        return

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    # Generate response
    prompt = "Write a haiku about artificial intelligence"
    if rank == 0:
        print(f"📝 Generating response for: {{prompt}}")

    start_time = time.time()
    try:
        response = generate(model, tokenizer, prompt, max_tokens=100)
        gen_time = time.time() - start_time

        tokens = len(tokenizer.encode(response))
        speed = tokens / gen_time if gen_time > 0 else 0

        # Show results from each node
        for i in range(size):
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))
            if rank == i:
                print(f"\n🎭 Node {{rank}} ({{hostname}}):")
                print(f"📝 {{response.strip()}}")
                print(f"⚡ {{speed:.1f}} tok/s ({{gen_time:.2f}}s)")
                print("-" * 40)
    except Exception as e:
        print(f"❌ Generation failed on rank {{rank}}: {{e}}")

    mx.eval(mx.distributed.all_sum(mx.array([1.0])))

    if rank == 0:
        print("🏁 Distributed prompting completed")

if __name__ == "__main__":
    main()
