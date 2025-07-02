#!/usr/bin/env bash
set -e

# Path to your conda setup script (may vary by system).
# Adjust if conda is installed in a different location.
CONDA_SETUP="/Users/alex/miniconda3/etc/profile.d/conda.sh"

# Name of the conda environment
CONDA_ENV="mlxdist"

# Path to your MLX project
PROJECT_PATH="/Users/zz/Documents/GitHub/mlx-dist-setup"

# If you haven’t already sourced your conda setup, do so:
if [ -f "$CONDA_SETUP" ]; then
  # Use "." instead of "source" for POSIX compatibility
  . "$CONDA_SETUP"
else
  echo "WARNING: Can't find conda setup script at $CONDA_SETUP"
  echo "         Make sure conda is properly set up before proceeding."
fi

echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Tell Open MPI to use the en0 interface for TCP connections
export OMPI_MCA_btl_tcp_if_include=en0

echo "Running distributed MLX job..."

mlx.launch \
  --hostfile "$PROJECT_PATH/hosts.json" \
  --backend mpi \
  "$PROJECT_PATH/pipeline_generate.py" \
  --prompt "What number is larger 6.9 or 6.11?" \
  --max-tokens 128 \
  --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx

echo "MLX run complete!"
