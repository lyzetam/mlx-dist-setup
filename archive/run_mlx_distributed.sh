#!/bin/bash
# Run MLX distributed across your Mac cluster

SCRIPT="${1:-test_mlx_dist.py}"
PROCESSES_PER_HOST="${2:-2}"
REMOTE_HOSTS=("mm1.local" "mm2.local")
REMOTE_USER="mm"

echo "🚀 MLX Distributed Cluster Setup Check"
echo "======================================"
echo "Target script: $SCRIPT"
echo "Processes per host: $PROCESSES_PER_HOST"
echo "Hosts: mbp.local, mm1.local, mm2.local"
echo ""

# Function to test connectivity
test_connectivity() {
    local host=$1
    echo -n "Testing $host... "
    
    # Test ping first
    if ! ping -c 1 -W 1000 $host >/dev/null 2>&1; then
        echo "❌ PING FAILED"
        return 1
    fi
    
    # Test SSH
    if ! ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_USER}@$host "echo 'OK'" >/dev/null 2>&1; then
        echo "❌ SSH FAILED"
        return 1
    fi
    
    # Test conda environment
    if ! ssh -o ConnectTimeout=3 ${REMOTE_USER}@$host "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlx-distributed && python -c 'import mlx.core as mx; print(\"MLX OK\")'" >/dev/null 2>&1; then
        echo "❌ MLX/CONDA FAILED"
        return 1
    fi
    
    echo "✅ OK"
    return 0
}

# Test connectivity to all remote hosts
echo "🔍 Testing connectivity to remote nodes..."
failed_hosts=()

for host in "${REMOTE_HOSTS[@]}"; do
    if ! test_connectivity $host; then
        failed_hosts+=($host)
    fi
done

echo ""

# Handle connectivity results
if [ ${#failed_hosts[@]} -eq 0 ]; then
    echo "✅ All nodes are reachable and ready!"
    echo ""
    echo "🚀 Starting distributed inference..."
    echo "Hosts: mbp.local, mm1.local, mm2.local"
    echo "Processes per host: $PROCESSES_PER_HOST"
    echo "Total processes: $((3 * PROCESSES_PER_HOST))"
    echo ""
    
    /Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi \
        --hosts mbp.local,mm1.local,mm2.local \
        --env "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlx-distributed" \
        -n "$PROCESSES_PER_HOST" \
        "$SCRIPT"
        
else
    echo "❌ Some nodes are not reachable: ${failed_hosts[*]}"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo ""
    
    for host in "${failed_hosts[@]}"; do
        echo "For $host:"
        echo "  1. Test SSH manually: ssh ${REMOTE_USER}@$host"
        echo "  2. Set up SSH keys: ssh-copy-id ${REMOTE_USER}@$host"
        echo "  3. Install MLX environment (see setup_guide.sh)"
        echo ""
    done
    
    echo "🎯 Alternative options:"
    echo ""
    echo "1. Run on localhost only (recommended):"
    echo "   ./run_mlx_local.sh $((3 * PROCESSES_PER_HOST)) $SCRIPT"
    echo ""
    echo "2. Run ring inference (high performance):"
    echo "   ./run_mlx_ring.sh $SCRIPT $((3 * PROCESSES_PER_HOST))"
    echo ""
    echo "3. Fix connectivity and try again"
    echo ""
    
    # Prompt user for action
    read -p "Choose option [1=localhost, 2=ring, 3=exit]: " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            echo "🚀 Running on localhost with $((3 * PROCESSES_PER_HOST)) processes..."
            ./run_mlx_local.sh $((3 * PROCESSES_PER_HOST)) "$SCRIPT"
            ;;
        2)
            echo "🚀 Running ring inference with $((3 * PROCESSES_PER_HOST)) processes..."
            ./run_mlx_ring.sh "$SCRIPT" $((3 * PROCESSES_PER_HOST))
            ;;
        3|*)
            echo "Exiting. Fix connectivity issues and try again."
            exit 1
            ;;
    esac
fi
