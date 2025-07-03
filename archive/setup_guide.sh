#!/bin/bash
# Complete setup guide for MLX distributed on remote nodes

echo "🎯 Complete Setup Guide for Remote MLX Nodes"
echo "============================================"
echo ""
echo "Follow these steps to set up mm1.local and mm2.local:"
echo ""

echo "STEP 1: Test Basic Connectivity"
echo "-------------------------------"
echo "Run these commands from your main machine (mbp):"
echo ""
echo "# Test ping connectivity"
echo "ping -c 3 mm1.local"
echo "ping -c 3 mm2.local"
echo ""
echo "# If ping fails, check network/DNS settings"
echo ""

echo "STEP 2: Set Up SSH Keys (if needed)"
echo "-----------------------------------"
echo "If you haven't set up passwordless SSH:"
echo ""
echo "# Generate SSH key (if you don't have one)"
echo "ssh-keygen -t rsa -b 4096"
echo ""
echo "# Copy SSH key to remote hosts"
echo "ssh-copy-id mm@mm1.local"
echo "ssh-copy-id mm@mm2.local"
echo ""
echo "# Test SSH connectivity"
echo "ssh mm@mm1.local 'echo SSH to mm1.local works'"
echo "ssh mm@mm2.local 'echo SSH to mm2.local works'"
echo ""

echo "STEP 3: Remote Node Setup"
echo "-------------------------"
echo "Log into EACH remote node and run these commands:"
echo ""
echo "## For mm1.local:"
echo "ssh mm@mm1.local"
echo ""
echo "## For mm2.local:"
echo "ssh mm@mm2.local"
echo ""
echo "## Then run these commands on EACH remote host:"
echo ""

cat << 'EOF'
# Check if conda is installed
which conda

# If conda is not found, install Miniconda:
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
# bash Miniconda3-latest-MacOSX-arm64.sh

# Remove any existing MLX environment
conda env remove -n mlx-distributed -y 2>/dev/null || true

# Create new environment with ARM64 support
CONDA_SUBDIR=osx-arm64 conda create -n mlx-distributed python=3.11 -y

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlx-distributed

# Configure for ARM64
conda config --env --set subdir osx-arm64

# Install MLX and dependencies
pip install mlx mlx-lm numpy

# Install MPI support
conda install -c conda-forge openmpi mpi4py -y

# Verify installation
python3 << 'PYTEST'
import mlx.core as mx
import socket
import sys

print(f"🖥️  Host: {socket.gethostname()}")
print(f"🐍 Python: {sys.version}")
print(f"🔧 MLX Metal: {mx.metal.is_available()}")

# Test GPU setup
mx.set_default_device(mx.gpu)
print(f"📱 Device: {mx.default_device()}")

# Test basic MLX operation
a = mx.random.uniform(shape=(100, 100))
b = mx.random.uniform(shape=(100, 100))
c = a @ b
mx.eval(c)
print(f"✅ MLX computation test: PASSED")
print(f"🚀 Node ready for distributed inference!")
PYTEST

# If the test above passes, the node is ready!
echo "✅ $(hostname) setup complete!"
EOF

echo ""
echo "STEP 4: Test Distributed Setup"
echo "------------------------------"
echo "After setting up BOTH remote nodes, test from your main machine:"
echo ""
echo "# Test basic distributed communication"
echo "./run_mlx_distributed.sh test_cluster_health.py"
echo ""
echo "# If that works, run the full inference"
echo "./run_mlx_distributed.sh working_dist_inference.py"
echo ""

echo "STEP 5: Troubleshooting"
echo "-----------------------"
echo "If distributed setup hangs:"
echo ""
echo "1. Check that all nodes have the same MLX version:"
echo "   ssh mm@mm1.local 'conda activate mlx-distributed && python -c \"import mlx; print(mlx.__version__)\"'"
echo "   ssh mm@mm2.local 'conda activate mlx-distributed && python -c \"import mlx; print(mlx.__version__)\"'"
echo ""
echo "2. Check MPI connectivity:"
echo "   ssh mm@mm1.local 'conda activate mlx-distributed && which mpirun'"
echo "   ssh mm@mm2.local 'conda activate mlx-distributed && which mpirun'"
echo ""
echo "3. Test SSH from main node to ensure passwordless access works"
echo ""
echo "4. Check firewalls and network connectivity between nodes"
echo ""

echo "🎉 SUMMARY"
echo "=========="
echo "1. Set up SSH keys for passwordless access"
echo "2. Install conda and MLX environment on each remote node"
echo "3. Test with: ./run_mlx_distributed.sh test_cluster_health.py"
echo "4. Run inference with: ./run_mlx_distributed.sh working_dist_inference.py"
