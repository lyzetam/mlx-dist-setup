#!/usr/bin/env bash

echo "==================================================="
echo "Testing MLX with Local Run (No Distribution)"
echo "==================================================="
echo

# Source the environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env_config.sh" ]; then
    . "${SCRIPT_DIR}/env_config.sh"
else
    echo "ERROR: env_config.sh not found"
    exit 1
fi

# Source conda
if [ -f "$CONDA_SETUP" ]; then
    . "$CONDA_SETUP"
else
    echo "ERROR: Cannot find conda setup"
    exit 1
fi

# Activate environment
echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Create a simple test script
echo "Creating local test script..."
cat > test_local_generate.py << 'EOF'
import mlx.core as mx
from mlx_lm import load, generate

# Simple test with a small model
model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
prompt = "Hello, how are you?"
max_tokens = 20

print(f"Loading model: {model_path}")
print(f"Prompt: {prompt}")
print("-" * 50)

try:
    model, tokenizer = load(model_path)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    print("Response:", response)
    print("-" * 50)
    print("✓ Local generation successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
EOF

# Run the test
echo
echo "Running local MLX test..."
python test_local_generate.py

# Clean up
rm -f test_local_generate.py

echo
echo "==================================================="
echo "Local test complete!"
echo
echo "If this worked, you can try the distributed version:"
echo "  ./run_mlx.sh"
echo "==================================================="