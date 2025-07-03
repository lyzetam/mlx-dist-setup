#!/bin/bash
# Run MLX with ring backend (optimized for Thunderbolt)

SCRIPT="${1:-test_mlx_dist.py}"

echo "Running MLX with ring backend (Thunderbolt optimized)"
echo "This should be faster for your Mac cluster"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend ring \
    --hosts mbp.local,mm1.local,mm2.local \
    -n 2 \
    --connections-per-ip 4 \
    --starting-port 5555 \
    "$SCRIPT"
