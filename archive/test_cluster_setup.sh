#!/bin/bash
# Test cluster connectivity and setup

echo "🔍 MLX Distributed Cluster Connectivity Test"
echo "==========================================="

REMOTE_HOSTS=("mm1.local" "mm2.local")
REMOTE_USER="mm"

echo ""
echo "1️⃣  Testing network connectivity..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -n "   Ping $host: "
    if ping -c 1 -W 1000 $host >/dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ FAILED"
    fi
done

echo ""
echo "2️⃣  Testing SSH connectivity..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -n "   SSH ${REMOTE_USER}@$host: "
    if timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes ${REMOTE_USER}@$host "echo 'OK'" >/dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ FAILED - Run: ssh-copy-id ${REMOTE_USER}@$host"
    fi
done

echo ""
echo "3️⃣  Testing conda on remote hosts..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -n "   Conda on $host: "
    if timeout 10 ssh -o ConnectTimeout=3 ${REMOTE_USER}@$host "which conda" >/dev/null 2>&1; then
        echo "✅ Found"
    else
        echo "❌ Not found - Install Anaconda/Miniconda on $host"
    fi
done

echo ""
echo "4️⃣  Manual setup commands for each remote host:"
echo ""
echo "   For mm1.local:"
echo "   ssh ${REMOTE_USER}@mm1.local"
echo ""
echo "   For mm2.local:"
echo "   ssh ${REMOTE_USER}@mm2.local"
echo ""
echo "   Then run these commands on EACH remote host:"
echo ""
cat << 'EOF'
   # Create MLX environment
   conda env remove -n mlx-distributed -y 2>/dev/null || true
   CONDA_SUBDIR=osx-arm64 conda create -n mlx-distributed python=3.11 -y
   
   # Activate and configure
   source ~/anaconda3/etc/profile.d/conda.sh
   conda activate mlx-distributed
   conda config --env --set subdir osx-arm64
   
   # Install packages
   pip install mlx mlx-lm numpy
   conda install -c conda-forge openmpi mpi4py -y
   
   # Test installation
   python3 -c "
import mlx.core as mx
import socket
print(f'✅ MLX working on {socket.gethostname()}')
print(f'   Metal: {mx.metal.is_available()}')
mx.set_default_device(mx.gpu)
print(f'   Device: {mx.default_device()}')
print('✅ Ready for distributed inference!')
"
EOF

echo ""
echo "5️⃣  After setup, test distributed with:"
echo "   ./run_mlx_distributed.sh test_cluster_health.py"
