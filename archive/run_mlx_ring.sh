#!/bin/bash
# Run MLX with ring backend (optimized for Thunderbolt)

SCRIPT="${1:-test_mlx_dist.py}"

echo "Running MLX with ring backend (Thunderbolt optimized)"
echo "This should be faster for your Mac cluster"
echo ""

# Resolve hostnames to IPs for ring backend
HOSTS_LIST="mbp.local,mm1.local,mm2.local"
IFS=',' read -ra HOSTS_ARR <<< "$HOSTS_LIST"
RESOLVED_HOSTS=()
for host in "${HOSTS_ARR[@]}"; do
  ip=$(ping -c1 "$host" | head -1 | sed -E 's/.*\(([^)]+)\).*/\1/')
  RESOLVED_HOSTS+=("$ip")
done
HOSTS_STR=$(IFS=,; echo "${RESOLVED_HOSTS[*]}")

# Launch with resolved IP addresses
/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend ring \
    --hosts $HOSTS_STR \
    -n 2 \
    --connections-per-ip 4 \
    --starting-port 5555 \
    "$SCRIPT"
