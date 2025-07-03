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

# Verify paths
if ! check_paths; then
    exit 1
fi

# Source conda setup
if [ -f "$CONDA_SETUP" ]; then
    . "$CONDA_SETUP"
else
    echo "ERROR: Cannot find conda setup script at $CONDA_SETUP"
    exit 1
fi

echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Tell Open MPI to use the configured network interface for TCP connections
export OMPI_MCA_btl_tcp_if_include="${MLX_NETWORK_INTERFACE}"

echo "Running distributed MLX job..."
echo "Using network interface: ${MLX_NETWORK_INTERFACE}"
echo "Model: ${MLX_MODEL}"
echo "Hosts file: ${MLX_HOSTS_FILE}"

mlx.launch \
  --hostfile "${MLX_HOSTS_FILE}" \
  --backend mpi \
  "${MLX_PROJECT_PATH}/pipeline_generate.py" \
  --prompt "${MLX_DEFAULT_PROMPT}" \
  --max-tokens ${MLX_MAX_TOKENS} \
  --model ${MLX_MODEL}

echo "MLX run complete!"
