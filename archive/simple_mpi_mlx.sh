#!/bin/bash
# Simple MLX distributed runner using MPI backend

SCRIPT="${1:-test_cluster_health.py}"
NP="${2:-6}"

echo "🚀 MLX Simple MPI Runner"
echo "Script: $SCRIPT"
echo "Processes: $NP"
echo ""

# Use mlx.launch with MPI backend on localhost
/Users/zz/anaconda3/envs/mlx-distributed/bin//Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch \
    --backend mpi \
    --hosts localhost \
    -n 2 \
    "$SCRIPT"
