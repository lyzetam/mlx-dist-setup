#!/usr/bin/env bash
# Removed set -e to prevent silent failures

# Source script to copy (the environment setup script)
SRC_SETUP_SCRIPT="./setup_mlxdist.sh"

# Source script to copy (the memory limit script)
SRC_MEMORY_SCRIPT="./set_mem_limit.sh"

# Source script to copy (the MLX run script)
SRC_MLX_SCRIPT="./run_mlx.sh"

# Source the env_config script
SRC_ENV_CONFIG="./env_config.sh"

# Source the pipeline script
SRC_PIPELINE="./pipeline_generate.py"

# Source the hosts.json file
SRC_HOSTS_JSON="./hosts.json"

# Destination directory on remote machines:
DEST_DIR="$(pwd)"

# Read hosts from hosts.json
HOSTS=$(jq -r '.[].ssh' ./hosts.json)
echo "Hosts from hosts.json:"
echo "$HOSTS" | sed 's/^/  - /'
echo

# Get the current machine's hostname and user info
CURRENT_HOST=$(hostname)
CURRENT_HOST_SHORT=$(hostname | cut -d. -f1)
CURRENT_USER=$(whoami)
echo "Current machine info:"
echo "  Hostname (full): $CURRENT_HOST"
echo "  Hostname (short): $CURRENT_HOST_SHORT"
echo "  Username: $CURRENT_USER"
echo

# Function to check if a host is the current machine
is_current_machine() {
    local test_host="$1"
    
    # Try to detect if this host refers to the current machine
    # Method 1: Check if we can run a command without SSH (local execution test)
    if timeout 2s ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=1 "$test_host" "test -d '$DEST_DIR'" 2>/dev/null; then
        # If the exact same directory exists at the same path, it's likely the same machine
        if [ -d "$DEST_DIR" ]; then
            local test_marker="/tmp/.mlx_host_test_$$"
            touch "$test_marker"
            if timeout 2s ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=1 "$test_host" "test -f '$test_marker'" 2>/dev/null; then
                rm -f "$test_marker"
                return 0  # This is the current machine
            fi
            rm -f "$test_marker"
        fi
    fi
    
    return 1  # Not the current machine
}

echo "Detecting which hosts are remote..."
echo

# Process each host
echo "$HOSTS" | while IFS= read -r host; do
  # Skip empty lines
  [ -z "$host" ] && continue
  
  echo "Processing: $host"
  
  # Check if this is the current machine
  if is_current_machine "$host"; then
    echo "  → Skipping (detected as current machine)"
    echo
    continue
  fi
  
  echo "  → Copying files to remote host..."
  
  # Ensure the destination directory exists on the remote machine
  if ! ssh "$host" "mkdir -p \"$DEST_DIR\"" 2>/dev/null; then
    echo "  ✗ Error: Failed to create directory on $host"
    echo "    (Check SSH access: ssh $host)"
    echo
    continue
  fi
  
  # Copy all necessary files
  echo "  → Copying setup script..."
  if ! scp -q "$SRC_SETUP_SCRIPT" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy setup script"
    continue
  fi
  
  echo "  → Copying memory script..."
  if ! scp -q "$SRC_MEMORY_SCRIPT" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy memory script"
    continue
  fi
  
  echo "  → Copying MLX run script..."
  if ! scp -q "$SRC_MLX_SCRIPT" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy MLX script"
    continue
  fi
  
  echo "  → Copying env_config.sh..."
  if ! scp -q "$SRC_ENV_CONFIG" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy env_config.sh"
    continue
  fi
  
  echo "  → Copying pipeline_generate.py..."
  if ! scp -q "$SRC_PIPELINE" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy pipeline_generate.py"
    continue
  fi
  
  echo "  → Copying hosts.json..."
  if ! scp -q "$SRC_HOSTS_JSON" "$host:$DEST_DIR/" 2>/dev/null; then
    echo "  ✗ Error: Failed to copy hosts.json"
    continue
  fi

  # Make scripts executable on remote host
  echo "  → Making scripts executable..."
  ssh "$host" "cd \"$DEST_DIR\" && chmod +x *.sh" 2>/dev/null
  
  echo "  ✓ All files copied successfully to $host"
  echo
done

echo "==================================================="
echo "Copy operation complete!"
echo
echo "Next steps for each remote machine:"
echo "1. SSH into each remote machine"
echo "2. Run: cd $DEST_DIR"
echo "3. Run: ./setup_mlxdist.sh"
echo "4. Run: ./set_mem_limit.sh (if on macOS)"
echo
echo "After setup on all machines:"
echo "Run from this machine: ./run_mlx.sh"
echo "==================================================="