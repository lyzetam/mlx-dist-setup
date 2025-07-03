#!/bin/bash
# Auto-deploy MLX distributed environment to remote nodes

REMOTE_HOSTS=("mm1.local" "mm2.local")
REMOTE_USER="mm"
LOCAL_ENV_PATH="/Users/zz/anaconda3/envs/mlx-distributed"

echo "🚀 MLX Distributed Auto-Deployment"
echo "=================================="

# Function to deploy to a single node
deploy_to_node() {
    local host=$1
    echo "📦 Deploying to $host..."

    # Test SSH first
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes ${REMOTE_USER}@$host "echo 'SSH OK'" >/dev/null 2>&1; then
        echo "❌ SSH to $host failed. Setting up SSH keys..."
        ssh-copy-id ${REMOTE_USER}@$host
        if [ $? -ne 0 ]; then
            echo "❌ Failed to setup SSH for $host"
            return 1
        fi
    fi

    # Copy and run setup script
    ssh ${REMOTE_USER}@$host 'bash -s' << 'EOF'
# Remove old environment
conda env remove -n mlx-distributed -y 2>/dev/null || true

# Create new environment with exact same packages
CONDA_SUBDIR=osx-arm64 conda create -n mlx-distributed python=3.11 -y

# Activate environment (try multiple conda locations)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/homebrew/etc/profile.d/conda.sh ]; then
    source /opt/homebrew/etc/profile.d/conda.sh
fi

conda activate mlx-distributed
conda config --env --set subdir osx-arm64

# Install exact same packages as local
pip install mlx mlx-lm numpy transformers
conda install -c conda-forge openmpi mpi4py -y

echo "✅ Environment setup complete on $(hostname)"
EOF

    if [ $? -eq 0 ]; then
        echo "✅ Successfully deployed to $host"
        return 0
    else
        echo "❌ Deployment failed to $host"
        return 1
    fi
}

# Deploy to all remote nodes
for host in "${REMOTE_HOSTS[@]}"; do
    deploy_to_node $host
done

echo ""
echo "🎯 Testing distributed health across all nodes..."
./run_mlx_distributed.sh test_cluster_health.py
