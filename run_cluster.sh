#!/bin/bash
# Run MLX Distributed on your cluster

HOSTS="mbp.local:2,mm1.local:2,mm2.local:2"
SCRIPT="${1:-test_dist.py}"

echo "Running MLX distributed across cluster..."
/opt/homebrew/bin/mpirun \
    --host "$HOSTS" \
    --mca btl_tcp_if_include en0 \
    /Users/zz/anaconda3/envs/mlx-distributed/bin/python mlx_dist_wrapper.py "$SCRIPT"
