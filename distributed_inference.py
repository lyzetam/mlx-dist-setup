
import time
import socket
import os
import mlx.core as mx
from mlx_lm import load, generate
from typing import List, Dict, Any, Optional


def setup_distributed_environment():
    """Initialize distributed MLX environment."""
    try:
        world = mx.distributed.init()
        rank = world.rank()
        size = world.size()
        hostname = socket.gethostname()
        pid = os.getpid()

        # Set GPU as default device
        mx.set_default_device(mx.gpu)

        return world, rank, size, hostname, pid
    except Exception as e:
        print(f"Error initializing distributed environment: {e}")
        raise


def load_model_distributed(model_name: str, rank: int, hostname: str) -> tuple:
    """Load model with distributed coordination and monitoring."""
    if rank == 0:
        print(f"\n=== Loading Model on All Nodes ===")
        print(f"Model: {model_name}")

    start_time = time.time()

    try:
        model, tokenizer = load(model_name)
        load_time = time.time() - start_time

        # Report loading time from each node
        print(f"[Rank {rank}/{hostname}] Model loaded in {load_time:.2f}s")

        # Synchronize after loading
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        if rank == 0:
            print("✓ All nodes have loaded the model successfully")

        return model, tokenizer

    except Exception as e:
        print(f"[Rank {rank}/{hostname}] Error loading model: {e}")
        raise


def create_diverse_prompts() -> List[str]:
    """Create a variety of prompts for distributed inference."""
    return [
        "Explain the benefits of distributed computing on Apple Silicon:",
        "What makes MLX framework special for machine learning?",
        "How does Metal Performance Shaders accelerate AI workloads?",
        "Compare CPU vs GPU performance for matrix operations:",
        "What are the advantages of running models locally vs cloud?",
        "Describe the future of edge AI computing:",
        "How do neural networks benefit from parallel processing?",
        "What optimization techniques work best for transformer models?",
        "Explain memory management in modern ML frameworks:",
        "How does quantization affect model performance and accuracy?"
    ]


def generate_distributed_responses(
    model, 
    tokenizer, 
    rank: int, 
    size: int, 
    hostname: str,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """Generate responses in distributed fashion with comprehensive monitoring."""

    prompts = create_diverse_prompts()

    # Each rank gets a different prompt
    prompt = prompts[rank % len(prompts)]

    # Create chat messages
    messages = [
        {"role": "system", "content": "You are an expert AI assistant specializing in distributed computing and machine learning. Provide technical, accurate responses."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Monitor memory before generation
        initial_memory = mx.metal.get_active_memory() / 1024**3 if mx.metal.is_available() else 0

        # Generate with timing
        start_time = time.time()

        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False  # Reduce noise in distributed setting
        )

        generation_time = time.time() - start_time
        final_memory = mx.metal.get_active_memory() / 1024**3 if mx.metal.is_available() else 0

        # Calculate metrics
        response_tokens = len(tokenizer.encode(response))
        tokens_per_second = response_tokens / generation_time if generation_time > 0 else 0

        return {
            "rank": rank,
            "hostname": hostname,
            "prompt": prompt,
            "response": response,
            "generation_time": generation_time,
            "tokens_generated": response_tokens,
            "tokens_per_second": tokens_per_second,
            "memory_used": final_memory - initial_memory,
            "prompt_tokens": len(tokenizer.encode(formatted_prompt))
        }

    except Exception as e:
        print(f"[Rank {rank}/{hostname}] Error during generation: {e}")
        raise


def main_distributed():
    """Main distributed inference function."""
    try:
        # Initialize distributed environment
        world, rank, size, hostname, pid = setup_distributed_environment()

        if rank == 0:
            print("=" * 60)
            print("🚀 MLX DISTRIBUTED INFERENCE ACROSS ALL NODES")
            print("=" * 60)
            print(f"Total processes: {size}")
            print(f"Expected nodes: mbp.local, mm1.local, mm2.local")
            print("=" * 60)

        # Report node status
        print(f"[Rank {rank}/{size}] Process {pid} on {hostname}")
        print(f"[Rank {rank}] GPU available: {mx.metal.is_available()}")
        print(f"[Rank {rank}] Device: {mx.default_device()}")

        # Synchronize before model loading
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        # Load model on all nodes
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        model, tokenizer = load_model_distributed(model_name, rank, hostname)

        if rank == 0:
            print(f"\n=== Generating Responses on {size} Processes ===")

        # Generate responses
        result = generate_distributed_responses(
            model, tokenizer, rank, size, hostname, max_tokens=150
        )

        # Collect and display results in order
        for i in range(size):
            # Synchronization barrier
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))

            if rank == i:
                print(f"\n📍 [Rank {result['rank']}/{result['hostname']}]")
                print(f"🔍 Prompt: {result['prompt']}")
                print(f"💬 Response: {result['response']}")
                print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
                print(f"🔢 Tokens: {result['tokens_generated']} ({result['tokens_per_second']:.1f} tok/s)")
                if result['memory_used'] > 0:
                    print(f"💾 GPU memory used: {result['memory_used']:.2f} GB")
                print("-" * 50)

        # Final synchronization and summary
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))

        if rank == 0:
            print(f"\n✅ DISTRIBUTED INFERENCE COMPLETE!")
            print(f"Successfully generated responses on {size} processes")
            print("=" * 60)

        return result

    except Exception as e:
        print(f"[Rank {rank if 'rank' in locals() else '?'}] Distributed inference failed: {e}")
        return None


if __name__ == "__main__":
    result = main_distributed()
