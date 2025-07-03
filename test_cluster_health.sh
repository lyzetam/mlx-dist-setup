#!/bin/bash
# Test MLX cluster health and connectivity

echo "=== MLX Cluster Health Check ==="
echo "Date: $(date)"
echo

# Configuration
HOSTS=("mbp.local" "mm1.local" "mm2.local")
PYTHON="/Users/zz/anaconda3/envs/mlx-distributed/bin/python"

# 1. Test SSH connectivity
echo "1. Testing SSH Connectivity..."
for host in "${HOSTS[@]}"; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $host "echo '   ✓ $host: SSH OK'" 2>/dev/null; then
        :
    else
        echo "   ✗ $host: SSH FAILED"
    fi
done

# 2. Test MLX installation
echo -e "\n2. Testing MLX Installation..."
for host in "${HOSTS[@]}"; do
    ssh $host "source ~/anaconda3/etc/profile.d/conda.sh && conda activate mlx-distributed && python -c 'import mlx.core as mx; print(f\"   ✓ {mx.default_device()} on '$host'\")" 2>/dev/null || echo "   ✗ $host: MLX not found"
done

# 3. Test MPI connectivity
echo -e "\n3. Testing MPI Connectivity..."
cat > test_mpi_cluster.py << 'EOF'
from mpi4py import MPI
import socket
comm = MPI.COMM_WORLD
print(f"   ✓ Rank {comm.rank}/{comm.size} on {socket.gethostname()}")
EOF

/opt/homebrew/bin/mpirun --host mbp.local,mm1.local,mm2.local \
    $PYTHON test_mpi_cluster.py 2>/dev/null || echo "   ✗ MPI test failed"

rm test_mpi_cluster.py

# 4. Quick distributed MLX test
echo -e "\n4. Testing Distributed MLX..."
$PYTHON/mlx.launch --backend mpi --hosts localhost -n 2 test_mlx_dist.py 2>&1 | grep -E "(✓|PASSED)" || echo "   ✗ MLX distributed test failed"

# 5. System resources
echo -e "\n5. System Resources:"
for host in "${HOSTS[@]}"; do
    echo -e "\n   $host:"
    ssh $host "echo '     CPU: '\$(sysctl -n hw.logicalcpu)' cores'; echo '     Memory: '\$(( \$(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))'GB'; echo '     GPU: Apple Silicon (Metal)'"
done

echo -e "\n=== Health Check Complete ==="