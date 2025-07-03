
import mlx.core as mx
from mlx_lm import load, generate
import socket
import time
import sys

def main():
    # Get process ID from command line
    process_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    mx.set_default_device(mx.gpu)

    print(f"🖥️  Process {process_id} @ {socket.gethostname()} starting...")

    # Load model
    start_time = time.time()
    try:
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        load_time = time.time() - start_time
        print(f"✅ Process {process_id}: Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"❌ Process {process_id}: Model loading failed - {e}")
        return

    # Generate response
    prompt = "Write a haiku about the beauty of parallel computing"

    # Add slight variation per process
    variations = [
        prompt,
        prompt + " in technological terms",
        prompt + " with creative metaphors"
    ]

    actual_prompt = variations[process_id % len(variations)]

    print(f"📝 Process {process_id}: Generating for '{actual_prompt[:50]}...'")

    start_time = time.time()
    try:
        response = generate(model, tokenizer, actual_prompt, max_tokens=100)
        gen_time = time.time() - start_time

        tokens = len(tokenizer.encode(response))
        speed = tokens / gen_time if gen_time > 0 else 0

        print(f"\n🎭 === RESPONSE FROM PROCESS {process_id} ===")
        print(f"📝 {response.strip()}")
        print(f"⚡ {speed:.1f} tok/s ({gen_time:.2f}s, {tokens} tokens)")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Process {process_id}: Generation failed - {e}")

if __name__ == "__main__":
    main()
