#!/bin/bash
# Run MLX distributed locally with MPI backend

NP="${1:-2}"
SCRIPT="${2:-test_mlx_dist.py}"

echo "Running MLX locally with $NP processes (MPI backend)..."
echo "Script: $SCRIPT"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi --hosts localhost -n "$NP" "$SCRIPT"
