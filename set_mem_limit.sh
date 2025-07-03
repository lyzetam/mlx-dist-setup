#!/usr/bin/env bash
set -e

# Source the environment configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env_config.sh" ]; then
    . "${SCRIPT_DIR}/env_config.sh"
else
    echo "ERROR: env_config.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is only for macOS (Apple Silicon)"
    exit 0
fi

# Check if memory limit is set
if [ -z "$MLX_MEMORY_LIMIT_MB" ] || [ "$MLX_MEMORY_LIMIT_MB" = "0" ]; then
    echo "No memory limit configured (MLX_MEMORY_LIMIT_MB=0 or not set)"
    exit 0
fi

echo "Setting iogpu.wired_limit_mb to ${MLX_MEMORY_LIMIT_MB} MB..."
sudo sysctl iogpu.wired_limit_mb=${MLX_MEMORY_LIMIT_MB}

echo "✓ Wired memory limit set to ${MLX_MEMORY_LIMIT_MB} MB!"
echo
echo "Note: This setting is temporary and will reset on reboot."
echo "To make it permanent, add to /etc/sysctl.conf:"
echo "  iogpu.wired_limit_mb=${MLX_MEMORY_LIMIT_MB}"
