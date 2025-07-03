import mlx.core as mx
from mlx_lm import load, generate
import time

def main():
    print("🤖 Testing MLX locally (no distributed)")
    print("=" * 40)
    
    mx.set_default_device(mx.gpu)
    
    # Load model
    start_time = time.time()
    try:
        print("📦 Loading Llama-3.2-1B model...")
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Generate response
    prompt = "Write a haiku about artificial intelligence"
    print(f"📝 Generating response for: {prompt}")
    
    start_time = time.time()
    try:
        response = generate(model, tokenizer, prompt, max_tokens=100)
        gen_time = time.time() - start_time
        
        tokens = len(tokenizer.encode(response))
        speed = tokens / gen_time if gen_time > 0 else 0
        
        print(f"\n🎭 Generated Response:")
        print(f"📝 {response.strip()}")
        print(f"⚡ {speed:.1f} tok/s ({gen_time:.2f}s)")
        print("-" * 40)
        print("✅ Local test completed successfully!")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
