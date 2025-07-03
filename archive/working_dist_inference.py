#!/bin/bash
# Enhanced MLX distributed runner with true multi-node support

SCRIPT="${1:-test_cluster_health.py}"
PROCESSES_PER_HOST="${2:-2}"
HOSTS="mbp.local,mm1.local,mm2.local"

echo "🚀 MLX Enhanced Distributed Runner"
echo "================================="
echo "Script: $SCRIPT"
echo "Hosts: $HOSTS"
echo "Processes per host: $PROCESSES_PER_HOST"
echo ""

# Test connectivity to all nodes first
echo "🔍 Testing node connectivity..."
failed_nodes=()
for host in ${HOSTS//,/ }; do
    if [[ "$host" == "mbp.local" ]]; then
        echo "✅ $host (local): OK"
        continue
    fi

    # Extract hostname without .local
    node_name=${host%%.local}
    if ping -c 1 -W 1000 $host >/dev/null 2>&1; then
        if ssh -o ConnectTimeout=3 -o BatchMode=yes mm@$host "conda activate mlx-distributed && python -c 'import mlx.core as mx; print(f\"MLX: {mx.metal.is_available()}\")" 2>/dev/null | grep -q "MLX: True"; then
            echo "✅ $host: OK (SSH + MLX working)"
        else
            echo "❌ $host: MLX environment issue"
            failed_nodes+=($host)
        fi
    else
        echo "❌ $host: Network unreachable"
        failed_nodes+=($host)
    fi
done

if [ ${#failed_nodes[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed nodes: ${failed_nodes[*]}"
    echo "💡 Run './deploy_to_nodes.sh' to auto-setup remote nodes"
    echo ""
    echo "🔄 Falling back to localhost with $((3 * PROCESSES_PER_HOST)) processes..."
    /Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi --hosts localhost -n $((3 * PROCESSES_PER_HOST)) "$SCRIPT"
else
    echo ""
    echo "✅ All nodes ready! Running true distributed..."
    # Use environment activation on remote nodes
    /Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi \
        --hosts $HOSTS \
        --env "conda activate mlx-distributed 2>/dev/null || source ~/.bashrc" \
        -n $PROCESSES_PER_HOST \
        "$SCRIPT"
fi
