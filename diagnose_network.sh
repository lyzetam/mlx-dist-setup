#!/bin/bash
# Network and firewall diagnostic script

echo "🔍 MLX Distributed Connectivity Diagnostics"
echo "==========================================="

REMOTE_HOSTS=("mm1.local" "mm2.local")
REMOTE_USER="mm"

echo ""
echo "1️⃣  Testing basic network connectivity..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -n "   Ping $host: "
    if ping -c 1 -W 2000 $host >/dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ FAILED - Check network/DNS"
    fi
done

echo ""
echo "2️⃣  Testing SSH connectivity..."
for host in "${REMOTE_HOSTS[@]}"; do
    echo -n "   SSH ${REMOTE_USER}@$host: "
    if timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes ${REMOTE_USER}@$host "echo 'OK'" 2>/dev/null; then
        echo "✅ OK"
    else
        echo "❌ FAILED"
        echo "      Try: ssh-copy-id ${REMOTE_USER}@$host"
        echo "      Or check: ssh ${REMOTE_USER}@$host"
    fi
done

echo ""
echo "3️⃣  Testing SSH with verbose output (first host)..."
echo "Running: ssh -vvv -o ConnectTimeout=10 ${REMOTE_USER}@${REMOTE_HOSTS[0]} 'echo test'"
timeout 15 ssh -vvv -o ConnectTimeout=10 ${REMOTE_USER}@${REMOTE_HOSTS[0]} 'echo test' 2>&1 | head -20

echo ""
echo "4️⃣  Checking firewall settings on local machine..."
echo "macOS Firewall status:"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

echo ""
echo "5️⃣  Testing MPI-specific ports..."
echo "Checking if common MPI ports are blocked..."

# Test some common MPI ports
for port in 22 2049 4000-4010; do
    echo -n "   Port $port to ${REMOTE_HOSTS[0]}: "
    if timeout 3 nc -z ${REMOTE_HOSTS[0]} 22 2>/dev/null; then
        echo "✅ Open"
    else
        echo "❌ Blocked/Closed"
    fi
done

echo ""
echo "6️⃣  Environment check..."
echo "Current MPI configuration:"
echo "   OMPI variables:"
env | grep OMPI || echo "   No OMPI variables set"

echo ""
echo "7️⃣  Quick fixes to try:"
echo ""
echo "   A) Disable macOS firewall temporarily:"
echo "      sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off"
echo ""
echo "   B) Allow specific applications through firewall:"
echo "      sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /opt/homebrew/bin/mpirun"
echo "      sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Users/zz/anaconda3/envs/mlx-distributed/bin/python3.11"
echo ""
echo "   C) Test with localhost only (works around network issues):"
echo "      ./run_mlx_local.sh 6 working_dist_inference.py"
echo ""
echo "   D) Use IP addresses instead of hostnames:"
echo "      Edit run_mlx_distributed.sh to use IP addresses"
echo ""
echo "   E) Set MPI to use specific network interface:"
echo "      export OMPI_MCA_btl_tcp_if_include=en0"
echo "      export OMPI_MCA_oob_tcp_if_include=en0"
