#!/usr/bin/env bash
# Removed set -e to prevent silent failures

# Source script to copy (the environment setup script)
SRC_SETUP_SCRIPT="./setup_mlxdist.sh"

# Source script to copy (the memory limit script)
SRC_MEMORY_SCRIPT="./set_mem_limit.sh"

# Source script to copy (the MLX run script)
SRC_MLX_SCRIPT="./run_mlx.sh"

# Destination directory on remote machines:
DEST_DIR="$(pwd)"

# Read hosts from hosts.json
HOSTS=$(jq -r '.[].ssh' ./hosts.json)
echo "Hosts: $HOSTS"

# Get the current machine's hostname (full and short version)
CURRENT_HOST=$(hostname)
CURRENT_HOST_SHORT=$(hostname | cut -d. -f1)
echo "Current host (full): $CURRENT_HOST"
echo "Current host (short): $CURRENT_HOST_SHORT"

echo "Copying scripts to remote machines..."

# Process each host using POSIX-compatible approach
echo "$HOSTS" | while IFS= read -r host; do
  # Skip empty lines
  [ -z "$host" ] && continue
  
  echo "Processing host: $host"
  # Skip copying if the host matches the current machine (either full or short hostname)
  if [ "$host" = "$CURRENT_HOST" ] || [ "$host" = "$CURRENT_HOST_SHORT" ]; then
    echo "Skipping copying to $host (current machine)"
    continue
  fi

  echo "--------------------------------------"
  echo "Copying scripts to $host ..."
  echo "Destination directory: $DEST_DIR"
  echo "Source setup script: $SRC_SETUP_SCRIPT"
  echo "Source memory script: $SRC_MEMORY_SCRIPT"
  echo "Source MLX script: $SRC_MLX_SCRIPT"

  # Ensure the destination directory exists on the remote machine:
  if ! ssh "$host" "mkdir -p \"$DEST_DIR\""; then
    echo "Error: Failed to create directory on $host"
    continue
  fi
  echo "Directory created successfully on $host"

  # Use scp to copy each file to the remote machine:
  if ! scp "$SRC_SETUP_SCRIPT" "$host:$DEST_DIR/"; then
    echo "Error: Failed to copy setup script to $host"
    continue
  fi
  echo "Setup script copied successfully to $host"
  
  if ! scp "$SRC_MEMORY_SCRIPT" "$host:$DEST_DIR/"; then
    echo "Error: Failed to copy memory script to $host"
    continue
  fi
  echo "Memory script copied successfully to $host"
  
  if ! scp "$SRC_MLX_SCRIPT" "$host:$DEST_DIR/"; then
    echo "Error: Failed to copy MLX script to $host"
    continue
  fi
  echo "MLX script copied successfully to $host"

  echo "Done copying scripts to $host:$DEST_DIR"
done

echo
echo "All copies complete!"
