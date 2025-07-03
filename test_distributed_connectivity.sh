#!/usr/bin/env bash

echo "==================================================="
echo "Testing Distributed MLX Connectivity"
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
    conda activate "$CONDA_ENV"
else
    echo "ERROR: Cannot find conda setup"
    exit 1
fi

# Test 1: SSH connectivity
echo "1. Testing SSH connectivity to all hosts..."
echo "----------------------------------------"
if [ -f "${MLX_HOSTS_FILE}" ]; then
    HOSTS=$(jq -r '.[].ssh' "${MLX_HOSTS_FILE}")
    CURRENT_HOST=$(hostname)
    CURRENT_HOST_SHORT=$(hostname -s)
    
    ALL_GOOD=true
    echo "$HOSTS" | while IFS= read -r host; do
        [ -z "$host" ] && continue
        
        if [ "$host" = "$CURRENT_HOST" ] || [ "$host" = "$CURRENT_HOST_SHORT" ]; then
            echo "  $host: (current machine, skipping)"
        else
            printf "  %-30s" "$host:"
            if ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo 'OK'" 2>/dev/null | grep -q "OK"; then
                echo "✓ SSH OK"
            else
                echo "✗ SSH FAILED"
                ALL_GOOD=false
            fi
        fi
    done
    
    if [ "$ALL_GOOD" = false ]; then
        echo
        echo "⚠ Fix SSH issues before proceeding:"
        echo "  1. Enable Remote Login on all Macs"
        echo "  2. Setup SSH keys: ssh-copy-id <hostname>"
    fi
else
    echo "ERROR: hosts.json not found"
    exit 1
fi

# Test 2: MPI basic test
echo
echo "2. Testing basic MPI functionality..."
echo "-----------------------------------"
export OMPI_MCA_btl_tcp_if_include="${MLX_NETWORK_INTERFACE}"

# Create test MPI script
cat > test_mpi.py << 'EOF'
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

print(f"Rank {rank}/{size} on {hostname}: MPI initialized successfully")

# Barrier to sync all processes
comm.Barrier()

if rank == 0:
    print(f"All {size} processes synchronized successfully")
EOF

echo "Running MPI test across all hosts..."
mpirun --hostfile "${MLX_HOSTS_FILE}" --map-by node python test_mpi.py 2>&1
MPI_RESULT=$?

if [ $MPI_RESULT -eq 0 ]; then
    echo "✓ MPI test successful"
else
    echo "✗ MPI test failed"
    echo
    echo "Common fixes:"
    echo "  1. Check firewall settings"
    echo "  2. Verify network interface: ${MLX_NETWORK_INTERFACE}"
    echo "  3. Ensure all nodes have the same conda environment"
fi

# Test 3: MLX distributed test
echo
echo "3. Testing MLX distributed setup..."
echo "---------------------------------"
cat > test_mlx_dist.py << 'EOF'
import mlx.core as mx

# Initialize distributed
group = mx.distributed.init(backend="mpi")
rank = group.rank()
size = group.size()

print(f"MLX Process {rank}/{size}: Distributed MLX initialized")

# Simple all-reduce test
test_array = mx.array([float(rank)])
result = mx.distributed.all_sum(test_array, stream=mx.cpu)
mx.eval(result)

if rank == 0:
    expected = sum(range(size))
    actual = result.item()
    if abs(actual - expected) < 0.001:
        print(f"✓ All-reduce test passed: sum={actual}")
    else:
        print(f"✗ All-reduce test failed: expected {expected}, got {actual}")
EOF

echo "Running MLX distributed test..."
mlx.launch --hostfile "${MLX_HOSTS_FILE}" --backend mpi test_mlx_dist.py 2>&1
MLX_RESULT=$?

if [ $MLX_RESULT -eq 0 ]; then
    echo "✓ MLX distributed test successful"
else
    echo "✗ MLX distributed test failed"
fi

# Clean up
rm -f test_mpi.py test_mlx_dist.py

echo
echo "==================================================="
echo "Connectivity test summary:"
echo
if [ $MPI_RESULT -eq 0 ] && [ $MLX_RESULT -eq 0 ]; then
    echo "✓ All tests passed! Your distributed setup is ready."
    echo "  You can now run: ./run_mlx.sh"
else
    echo "✗ Some tests failed. Please fix the issues above."
fi
echo "==================================================="