#!/bin/bash
# Manual setup for remote nodes - step by step

echo "=== Manual Remote Setup Guide ==="
echo
echo "Since automated setup is having issues, here's a manual approach."
echo "Run these commands on each remote node (mm1.local and mm2.local):"
echo
echo "Step 1: SSH into each node"
echo "   ssh mm@mm1.local  # or mm2.local"
echo
echo "Step 2: Set up environment (copy and paste these commands):"
cat << 'MANUAL_COMMANDS'

# Fix PATH first
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$HOME/miniconda3/bin:$PATH"

# Check if conda works
~/miniconda3/bin/conda --version

# If conda works, continue:
# Create environment
~/miniconda3/bin/conda create -n mlx-distributed python=3.11 -y

# Activate it
source ~/miniconda3/bin/activate mlx-distributed

# Install packages
pip install --upgrade pip
pip install mlx mlx-lm numpy transformers huggingface-hub

# Install mpi4py
if [ -f /opt/homebrew/bin/mpicc ]; then
    MPICC=/opt/homebrew/bin/mpicc pip install mpi4py --no-binary mpi4py
else
    pip install mpi4py
fi

# Create working directory
mkdir -p ~/mlx_distributed
cd ~/mlx_distributed

# Test it works
python -c "import mlx.core as mx; print(f'MLX ready: {mx.default_device()}')"

MANUAL_COMMANDS

echo
echo "Step 3: Copy the scripts from your main node"
echo "   From your main machine (mbp), run:"
echo "   scp *.py *.sh mlx_hostfile.txt mm@mm1.local:~/mlx_distributed/"
echo "   scp *.py *.sh mlx_hostfile.txt mm@mm2.local:~/mlx_distributed/"
echo
echo "Step 4: Test distributed setup"
echo "   ./run_mlx_distributed.sh test_mlx_dist.py"
echo
echo "=== End Manual Setup ==="

# Let's at least copy the files for you
echo
read -p "Should I copy the files to the remote nodes now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Copying files..."
    for host in "mm@mm1.local" "mm@mm2.local"; do
        echo "  Copying to $host..."
        ssh $host "mkdir -p ~/mlx_distributed"
        scp *.py *.sh mlx_hostfile.txt $host:~/mlx_distributed/ 2>/dev/null
        echo "  ✓ Files copied to $host"
    done
fi