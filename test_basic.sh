#!/bin/bash
# Test basic distributed functionality without full setup

echo "=== Testing Basic Distributed Setup ==="
echo

# Test 1: Local distributed (should work)
echo "1. Testing local distributed (2 processes on this machine)..."
if [ -f test_mlx_dist.py ]; then
    ./run_mlx_local.sh 2 test_mlx_dist.py
else
    echo "   Creating simple test..."
    cat > simple_test.py << 'EOF'
import mlx.core as mx
world = mx.distributed.init()
rank = world.rank()
size = world.size()
print(f"Process {rank}/{size}: Ready!")
if size > 1:
    data = mx.array([float(rank)])
    result = mx.distributed.all_sum(data)
    mx.eval(result)
    if rank == 0:
        print(f"Sum: {result.item()} (expected: {sum(range(size))})")
EOF
    ./run_mlx_local.sh 2 simple_test.py
    rm simple_test.py
fi

echo -e "\n2. Checking SSH access to remote nodes..."
for host in "mm1.local" "mm2.local"; do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes mm@$host "echo '   ✓ SSH to $host: OK'" 2>/dev/null; then
        # Check if Python exists
        ssh mm@$host "which python3 || which python || echo '   ✗ No Python found on $host'" 2>/dev/null
    else
        echo "   ✗ Cannot SSH to $host"
        echo "     Fix: ssh-copy-id mm@$host"
    fi
done

echo -e "\n3. Testing if MPI can reach remote nodes..."
# Create minimal test
cat > mpi_basic.py << 'EOF'
import socket
print(f"Hello from {socket.gethostname()}")
EOF

# Try with system Python first
echo "   Using system Python:"
/opt/homebrew/bin/mpirun --host mbp.local,mm1.local,mm2.local \
    -np 3 \
    --mca btl_tcp_if_exclude lo0 \
    --mca plm_base_verbose 1 \
    python3 mpi_basic.py 2>&1 | grep -E "(Hello from|unable to reach|Connection refused)" | head -10

rm mpi_basic.py

echo -e "\n=== Summary ==="
echo "If you see:"
echo "  - 'Hello from mbp': Local MPI works"
echo "  - 'Connection refused' or 'unable to reach': SSH/firewall issue" 
echo "  - 'Hello from mm1/mm2': Remote nodes are accessible!"
echo
echo "Next steps:"
echo "1. If SSH fails: ssh-copy-id mm@mm1.local && ssh-copy-id mm@mm2.local"
echo "2. If remote Python missing: Use manual_setup_remote.sh"
echo "3. If all works: The distributed setup should work!"