#!/bin/bash
# Run MLX using hostfile (MPI backend)

SCRIPT="${1:-test_mlx_dist.py}"
HOSTFILE="${2:-mlx_hostfile.txt}"

echo "Running MLX with hostfile (MPI backend)"
echo "Hostfile: $HOSTFILE"
echo "Script: $SCRIPT"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi --hostfile "$HOSTFILE" "$SCRIPT"
