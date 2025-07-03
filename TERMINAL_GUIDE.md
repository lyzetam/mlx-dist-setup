# 🚀 MLX Distributed Inference - Terminal Commands

## Quick Start Commands

### 1. Basic Distributed Inference
```bash
cd /Users/zz/Documents/GitHub/mlx-dist-setup

# Run with default prompt
./run_terminal.sh

# Run with custom prompt
./run_terminal.sh "Write a creative story about robots"
```

### 2. Direct MLX Launch Commands

#### Run the main distributed script:
```bash
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch \
  --backend mpi \
  --hosts mbp.local,mm1.local,mm2.local \
  -n 3 \
  distributed_inference.py
```

#### Run custom prompting:
```bash
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch \
  --backend mpi \
  --hosts mbp.local,mm1.local,mm2.local \
  -n 3 \
  custom_prompt.py
```

#### Run performance test:
```bash
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch \
  --backend mpi \
  --hosts mbp.local,mm1.local,mm2.local \
  -n 3 \
  performance_test.py
```

### 3. Local Multi-Process (if distributed fails)
```bash
# Run 3 parallel processes locally
python local_mlx_parallel.py 0 &
python local_mlx_parallel.py 1 &
python local_mlx_parallel.py 2 &
wait
```

### 4. Test Individual Scripts

#### Test MLX locally:
```bash
python test_local.py
```

#### Test SSH connectivity:
```bash
ssh mm1.local echo "Connection OK"
ssh mm2.local echo "Connection OK"
```

#### Test GPU on all nodes:
```bash
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch \
  --backend mpi \
  --hosts mbp.local,mm1.local,mm2.local \
  -n 3 \
  gpu_test.py
```

## Troubleshooting Commands

### Check MLX installation:
```bash
conda activate mlx-distributed
python -c "import mlx.core; print(f'MLX {mlx.core.__version__} ready!')"
```

### Check remote MLX:
```bash
ssh mm1.local "~/miniconda3/bin/conda run -n mlx-distributed python -c 'import mlx.core; print(mlx.core.__version__)'"
ssh mm2.local "~/miniconda3/bin/conda run -n mlx-distributed python -c 'import mlx.core; print(mlx.core.__version__)'"
```

### Monitor processes:
```bash
# Watch GPU usage
while true; do
  echo "=== $(date) ==="
  ssh mm1.local "ps aux | grep python"
  ssh mm2.local "ps aux | grep python"
  sleep 5
done
```

## Environment Variables

Set these for easier commands:
```bash
export MLX_LAUNCH="/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch"
export CLUSTER_HOSTS="mbp.local,mm1.local,mm2.local"
export MLX_PROJECT="/Users/zz/Documents/GitHub/mlx-dist-setup"

# Then use:
cd $MLX_PROJECT
$MLX_LAUNCH --backend mpi --hosts $CLUSTER_HOSTS -n 3 distributed_inference.py
```

## Create Custom Prompts

### Edit existing scripts:
```bash
# Edit the prompt in custom_prompt.py
nano custom_prompt.py

# Or create a quick one-liner
echo 'Write a haiku about machine learning' > /tmp/prompt.txt
```

### Quick custom script:
```bash
cat > quick_inference.py << 'EOF'
import mlx.core as mx
from mlx_lm import load, generate
import socket

def main():
    world = mx.distributed.init()
    rank = world.rank()
    hostname = socket.gethostname()
    
    mx.set_default_device(mx.gpu)
    
    if rank == 0:
        print("🚀 Quick MLX Inference")
    
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    prompt = "Explain quantum computing in simple terms"
    response = generate(model, tokenizer, prompt, max_tokens=100)
    
    print(f"Node {rank} ({hostname}): {response.strip()}")

if __name__ == "__main__":
    main()
EOF

# Run it
$MLX_LAUNCH --backend mpi --hosts $CLUSTER_HOSTS -n 3 quick_inference.py
```

## Performance Tips

- Add `--verbose` flag to see detailed MPI output
- Use `timeout 300` to limit execution time
- Check network latency: `ping mm1.local && ping mm2.local`
- Monitor GPU memory: `python -c "import mlx.core as mx; print(mx.metal.get_memory_info())"`
