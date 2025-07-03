#!/bin/bash
# Deploy MLX distributed environment to remote nodes

echo "=== MLX Distributed Deployment Script ==="
echo

# Configuration
CONDA_ENV="mlx-distributed"
LOCAL_DIR=$(pwd)
REMOTE_HOSTS=("mm@mm1.local" "mm@mm2.local")

# Step 1: Export conda environment
echo "1. Exporting conda environment..."
conda env export -n $CONDA_ENV > mlx_distributed_env.yml
echo "✓ Environment exported to mlx_distributed_env.yml"

# Step 2: Create deployment package
echo -e "\n2. Creating deployment package..."
tar -czf mlx_dist_deploy.tar.gz \
    *.py \
    *.sh \
    mlx_hostfile.txt \
    mlx_distributed_env.yml \
    --exclude="*.ipynb" \
    --exclude="__pycache__"
echo "✓ Package created: mlx_dist_deploy.tar.gz"

# Step 3: Deploy to each remote host
for host in "${REMOTE_HOSTS[@]}"; do
    echo -e "\n3. Deploying to $host..."
    
    # Copy deployment package
    echo "   - Copying files..."
    scp mlx_dist_deploy.tar.gz $host:~/
    
    # Create setup script for remote host
    ssh $host 'bash -s' << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

echo "   - Setting up on $(hostname)..."

# Create working directory
mkdir -p ~/mlx_distributed
cd ~/mlx_distributed

# Extract files
tar -xzf ~/mlx_dist_deploy.tar.gz

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "   ✗ Conda not found! Please install Anaconda/Miniconda first."
    exit 1
fi

# Initialize conda for bash (if needed)
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh || true

# Remove existing environment if it exists
conda env remove -n mlx-distributed -y 2>/dev/null || true

# Create conda environment from file
echo "   - Creating conda environment (this may take a few minutes)..."
conda env create -f mlx_distributed_env.yml

# Activate and verify
conda activate mlx-distributed

# Install mpi4py with Homebrew MPI (if available)
if [ -f /opt/homebrew/bin/mpicc ]; then
    echo "   - Installing mpi4py with Homebrew MPI..."
    MPICC=/opt/homebrew/bin/mpicc pip install mpi4py --no-cache-dir --no-binary mpi4py
else
    echo "   - Warning: Homebrew MPI not found, using system MPI"
fi

echo "   ✓ Setup complete on $(hostname)"
REMOTE_SCRIPT

done

# Step 4: Verify deployment
echo -e "\n4. Verifying deployment..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -e "\n   Checking $host..."
    ssh $host 'cd ~/mlx_distributed && source ~/anaconda3/etc/profile.d/conda.sh && conda activate mlx-distributed && python -c "import mlx.core as mx; print(f\"MLX ready on {mx.default_device()}\")"' || echo "   ✗ Verification failed on $host"
done

# Clean up
rm mlx_dist_deploy.tar.gz mlx_distributed_env.yml

echo -e "\n=== Deployment Complete ==="
echo "Remote nodes should now have:"
echo "  - MLX distributed environment"
echo "  - All test scripts"
echo "  - Launch scripts"
echo
echo "Next: Test distributed inference across all nodes!"