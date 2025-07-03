#!/bin/bash
# Quick deployment using pip instead of conda export

echo "=== Quick MLX Deployment to Remote Nodes ==="
echo

REMOTE_HOSTS=("mm@mm1.local" "mm@mm2.local")
PYTHON_VERSION="3.11"

# Create deployment package with scripts only
echo "1. Creating deployment package..."
tar -czf mlx_scripts.tar.gz *.py *.sh mlx_hostfile.txt 2>/dev/null
echo "✓ Scripts packaged"

for host in "${REMOTE_HOSTS[@]}"; do
    echo -e "\n2. Setting up $host..."
    
    # Copy scripts
    scp mlx_scripts.tar.gz $host:~/
    
    # Remote setup
    ssh $host << 'REMOTE_SETUP'
#!/bin/bash
set -e

echo "   Finding conda installation..."
# Find conda
CONDA_PATH=""
for path in ~/miniconda3 ~/anaconda3 /opt/homebrew/Caskroom/miniconda/base; do
    if [ -f "$path/etc/profile.d/conda.sh" ]; then
        CONDA_PATH="$path"
        break
    fi
done

if [ -z "$CONDA_PATH" ]; then
    echo "   ✗ Conda not found! Please run ./install_conda_remote.sh first"
    exit 1
fi

source "$CONDA_PATH/etc/profile.d/conda.sh"
echo "   ✓ Found conda at $CONDA_PATH"

# Create fresh environment
echo "   Creating Python 3.11 environment..."
conda create -n mlx-distributed python=3.11 -y

# Activate
conda activate mlx-distributed

# Install packages via pip (faster than conda for these)
echo "   Installing MLX and dependencies..."
pip install --upgrade pip

# Core packages
pip install mlx mlx-lm numpy

# MPI support
if [ -f /opt/homebrew/bin/mpicc ]; then
    MPICC=/opt/homebrew/bin/mpicc CC=/opt/homebrew/bin/mpicc pip install mpi4py --no-binary mpi4py
else
    pip install mpi4py
fi

# Additional utilities
pip install transformers huggingface-hub

# Extract scripts
mkdir -p ~/mlx_distributed
cd ~/mlx_distributed
tar -xzf ~/mlx_scripts.tar.gz
rm ~/mlx_scripts.tar.gz

# Test installation
echo "   Testing MLX..."
python -c "import mlx.core as mx; print(f'   ✓ MLX ready: {mx.default_device()}')"

echo "   ✓ Setup complete on $(hostname)!"
REMOTE_SETUP

done

# Cleanup
rm mlx_scripts.tar.gz

echo -e "\n=== Quick Deployment Complete ==="
echo "Test with: ./run_mlx_distributed.sh test_mlx_dist.py"