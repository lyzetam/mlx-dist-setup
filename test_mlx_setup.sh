#!/usr/bin/env bash

echo "==================================================="
echo "Testing MLX Environment Setup"
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
if conda activate "$CONDA_ENV" 2>/dev/null; then
    echo "✓ Environment activated"
else
    echo "✗ Failed to activate environment"
    exit 1
fi

# Test imports
echo
echo "Testing Python imports..."
python << EOF
import sys
print(f"Python: {sys.version}")

try:
    import mlx
    print("✓ MLX imported successfully")
except ImportError as e:
    print("✗ MLX import failed:", e)

try:
    import mlx_lm
    print("✓ mlx-lm imported successfully")
except ImportError as e:
    print("✗ mlx-lm import failed:", e)

try:
    from mpi4py import MPI
    print("✓ mpi4py imported successfully")
    print(f"  MPI Version: {MPI.Get_version()}")
except ImportError as e:
    print("✗ mpi4py import failed:", e)
EOF

# Test MPI
echo
echo "Testing MPI setup..."
if which mpirun >/dev/null 2>&1; then
    echo "✓ mpirun found at: $(which mpirun)"
    mpirun --version | head -n1
else
    echo "✗ mpirun not found"
fi

# Test local MPI run
echo
echo "Testing local MPI execution..."
mpirun -np 2 python -c "from mpi4py import MPI; print(f'Process {MPI.COMM_WORLD.rank} of {MPI.COMM_WORLD.size}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Local MPI execution successful"
else
    echo "✗ Local MPI execution failed"
fi

# Check hosts
echo
echo "Checking hosts configuration..."
if [ -f "${MLX_HOSTS_FILE}" ]; then
    echo "Hosts in ${MLX_HOSTS_FILE}:"
    jq -r '.[].ssh' "${MLX_HOSTS_FILE}" | while read -r host; do
        if [ "$host" = "$(hostname)" ] || [ "$host" = "$(hostname -s)" ]; then
            echo "  - $host (current machine)"
        else
            echo "  - $host"
        fi
    done
else
    echo "✗ hosts.json not found"
fi

# Check network interface
echo
echo "Network configuration:"
echo "  Configured interface: ${MLX_NETWORK_INTERFACE}"
echo "  Available interfaces:"
ifconfig | grep -B2 "inet " | grep -v "127.0.0.1" | grep "^[a-z]" | cut -d: -f1 | sort -u | sed 's/^/    - /'

echo
echo "==================================================="
echo "Test complete!"
echo "If all tests pass, you can run: ./run_mlx.sh"
echo "==================================================="