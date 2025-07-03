#!/bin/bash

# 🚀 MLX Distributed Inference - Terminal Runner
# Usage: ./run_terminal.sh [prompt]
# Example: ./run_terminal.sh "Write a haiku about AI"

echo "🚀 MLX Distributed Inference - Terminal Mode"
echo "=============================================="

# Configuration
CLUSTER_HOSTS="mbp.local,mm1.local,mm2.local"
NUM_NODES=3
USER="zz"
CONDA_ENV="mlx-distributed"
MLX_LAUNCH="/Users/$USER/anaconda3/envs/$CONDA_ENV/bin/mlx.launch"

# Get prompt from command line or use default
if [ "$1" ]; then
    PROMPT="$1"
else
    PROMPT="Write a haiku about distributed computing across multiple Macs"
fi

echo "🎤 Prompt: $PROMPT"
echo "🌐 Nodes: $CLUSTER_HOSTS"
echo "📊 Processes: $NUM_NODES"
echo ""

# Check if MLX launcher exists
if [ ! -f "$MLX_LAUNCH" ]; then
    echo "❌ MLX launcher not found at: $MLX_LAUNCH"
    echo "💡 Try running the notebook setup first"
    exit 1
fi

# Create a temporary distributed script with the custom prompt
TEMP_SCRIPT="temp_distributed_$(date +%s).py"

cat > "$TEMP_SCRIPT" << EOF
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
        print("🌐 TERMINAL MLX DISTRIBUTED INFERENCE")
        print(f"📊 Nodes: {size}")
        print("=" * 50)
    
    # Load model
    print(f"[Rank {rank}@{hostname}] Loading model...")
    start_time = time.time()
    try:
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        load_time = time.time() - start_time
        print(f"[Rank {rank}@{hostname}] ✅ Model loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"[Rank {rank}@{hostname}] ❌ Model loading failed: {e}")
        return
    
    # Synchronize
    mx.eval(mx.distributed.all_sum(mx.array([1.0])))
    
    # Use the provided prompt
    prompt = "$PROMPT"
    
    if rank == 0:
        print(f"📝 Generating responses for: {prompt}")
    
    # Generate response
    start_time = time.time()
    try:
        response = generate(model, tokenizer, prompt, max_tokens=120)
        gen_time = time.time() - start_time
        
        tokens = len(tokenizer.encode(response))
        speed = tokens / gen_time if gen_time > 0 else 0
        
        # Display results from each node
        for i in range(size):
            mx.eval(mx.distributed.all_sum(mx.array([1.0])))
            
            if rank == i:
                print(f"\\n🖥️  Node {rank} ({hostname}):")
                print(f"🎨 {response.strip()}")
                print(f"⚡ {speed:.1f} tok/s ({gen_time:.2f}s, {tokens} tokens)")
                print("-" * 50)
        
        mx.eval(mx.distributed.all_sum(mx.array([1.0])))
        
        if rank == 0:
            print("\\n✅ Distributed inference complete!")
            
    except Exception as e:
        print(f"[Rank {rank}@{hostname}] ❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
EOF

echo "⏳ Starting distributed inference..."
echo "🚀 Command: mlx.launch --backend mpi --hosts $CLUSTER_HOSTS -n $NUM_NODES"
echo ""

# Run the distributed inference
start_time=$(date +%s)
"$MLX_LAUNCH" --backend mpi --hosts "$CLUSTER_HOSTS" -n "$NUM_NODES" "$TEMP_SCRIPT"
end_time=$(date +%s)

execution_time=$((end_time - start_time))
echo ""
echo "⏱️  Total execution time: ${execution_time}s"
echo "🎯 Distributed inference complete!"

# Cleanup
rm -f "$TEMP_SCRIPT"
