#!/bin/bash
# Quick MLX distributed test bypassing connectivity checks

SCRIPT="${1:-test_cluster_health.py}"
PROCESSES_PER_HOST="${2:-2}"

echo "🚀 Quick MLX Distributed Test"
echo "============================="
echo "Script: $SCRIPT"
echo "Processes per host: $PROCESSES_PER_HOST"
echo "Hosts: mbp.local, mm1.local, mm2.local"
echo ""

echo "🎯 Running distributed MLX directly..."
echo "Command: mlx.launch --backend mpi --hosts mbp.local,mm1.local,mm2.local -n $PROCESSES_PER_HOST $SCRIPT"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi \
    --hosts mbp.local,mm1.local,mm2.local \
    -n "$PROCESSES_PER_HOST" \
    "$SCRIPT"

echo ""
echo "🎉 Distributed test complete!"
