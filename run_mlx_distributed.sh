#!/bin/bash
# Run MLX distributed across your Mac cluster

SCRIPT="${1:-test_mlx_dist.py}"
PROCESSES_PER_HOST="${2:-2}"

echo "Running MLX distributed (MPI backend)"
echo "Hosts: mbp.local, mm1.local, mm2.local"
echo "Processes per host: $PROCESSES_PER_HOST"
echo "Script: $SCRIPT"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi \
    --hosts mbp.local,mm1.local,mm2.local \
    -n "$PROCESSES_PER_HOST" \
    "$SCRIPT"
