#!/bin/bash
# Quick firewall and network fixes for MLX distributed

echo "🚨 Quick Fixes for MLX Distributed Connectivity Issues"
echo "====================================================="

echo ""
echo "🔧 Fix 1: Disable macOS Firewall (Temporary)"
echo "---------------------------------------------"
echo "This will temporarily disable the macOS firewall to test if it's blocking MPI:"
echo ""
read -p "Disable firewall temporarily? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Disabling firewall..."
    sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
    echo "✅ Firewall disabled. Test your distributed setup now."
    echo "⚠️  Remember to re-enable it later with:"
    echo "   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on"
else
    echo "Firewall left unchanged."
fi

echo ""
echo "🔧 Fix 2: Set MPI Network Interface"
echo "-----------------------------------"
echo "Setting MPI to use the primary network interface..."

# Get the primary network interface
PRIMARY_INTERFACE=$(route get default | grep interface | awk '{print $2}')
echo "Primary interface detected: $PRIMARY_INTERFACE"

export OMPI_MCA_btl_tcp_if_include=$PRIMARY_INTERFACE
export OMPI_MCA_oob_tcp_if_include=$PRIMARY_INTERFACE

echo "Set MPI environment variables:"
echo "   OMPI_MCA_btl_tcp_if_include=$PRIMARY_INTERFACE"
echo "   OMPI_MCA_oob_tcp_if_include=$PRIMARY_INTERFACE"

# Make it persistent
echo "export OMPI_MCA_btl_tcp_if_include=$PRIMARY_INTERFACE" >> ~/.zshrc
echo "export OMPI_MCA_oob_tcp_if_include=$PRIMARY_INTERFACE" >> ~/.zshrc

echo ""
echo "🔧 Fix 3: Test Basic SSH Connectivity"
echo "------------------------------------"
REMOTE_HOSTS=("mm1.local" "mm2.local")
REMOTE_USER="mm"

for host in "${REMOTE_HOSTS[@]}"; do
    echo "Testing SSH to ${REMOTE_USER}@$host..."
    if timeout 10 ssh -o ConnectTimeout=5 ${REMOTE_USER}@$host "echo 'SSH to $host works'; hostname; date" 2>/dev/null; then
        echo "✅ SSH to $host: OK"
    else
        echo "❌ SSH to $host: FAILED"
        echo "   Fix: ssh-copy-id ${REMOTE_USER}@$host"
        echo "   Or test manually: ssh ${REMOTE_USER}@$host"
    fi
done

echo ""
echo "🔧 Fix 4: Create Localhost-Only Version"
echo "--------------------------------------"
echo "Creating a localhost-only version that avoids network issues..."

cat > run_mlx_cluster.sh << 'EOF'
#!/bin/bash
# Run MLX with multiple processes on localhost (no network issues)

SCRIPT="${1:-working_dist_inference.py}"
PROCESSES="${2:-6}"

echo "Running MLX distributed on localhost only"
echo "Processes: $PROCESSES"
echo "Script: $SCRIPT"
echo ""

/Users/zz/anaconda3/envs/mlx-distributed/bin/mlx.launch --backend mpi --hosts localhost -n "$PROCESSES" "$SCRIPT"
EOF

chmod +x run_mlx_cluster.sh

echo "✅ Created run_mlx_cluster.sh for localhost-only testing"

echo ""
echo "🧪 Test Commands"
echo "==============="
echo ""
echo "1. Test localhost only (should work):"
echo "   ./run_mlx_cluster.sh working_dist_inference.py 6"
echo ""
echo "2. Test distributed (after fixes):"
echo "   ./run_mlx_distributed.sh working_dist_inference.py"
echo ""
echo "3. If still failing, check individual SSH:"
echo "   ssh mm@mm1.local"
echo "   ssh mm@mm2.local"

echo ""
echo "🔍 Additional Troubleshooting"
echo "=============================="
echo ""
echo "If issues persist:"
echo "1. Check if remote hosts have same conda environment"
echo "2. Verify MLX is installed on remote hosts"
echo "3. Test with IP addresses instead of hostnames"
echo "4. Check for VPN or security software interference"
