#!/bin/bash
# Run MLX distributed across your cluster

SCRIPT="${1:-test_mlx_dist.py}"

# Option 1: Using comma-separated hosts with repetition
echo "=== Running distributed with 2 processes per host ==="
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --hosts mbp.local,mm1.local,mm2.local -n 2 "$SCRIPT"
