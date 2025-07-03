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

# Since mlx-env already exists, just verify it
echo "Checking existing MLX environment..."
echo "Project path: ${MLX_PROJECT_PATH}"
echo "Conda base: ${CONDA_BASE_PATH}"
echo "Conda env: ${CONDA_ENV}"

# Change to project directory
cd "${MLX_PROJECT_PATH}"

# Source conda
if [ -f "${CONDA_SETUP}" ]; then
    . "${CONDA_SETUP}"
else
    echo "ERROR: Cannot find conda activation script at ${CONDA_SETUP}"
    exit 1
fi

# Check if environment exists
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "✓ Conda environment '${CONDA_ENV}' exists"
    
    # Activate and verify packages
    conda activate "${CONDA_ENV}"
    
    echo "Verifying installed packages..."
    
    # Check for mlx-lm
    if python -c "import mlx_lm" 2>/dev/null; then
        echo "✓ mlx-lm is installed"
    else
        echo "Installing mlx-lm..."
        pip install -U mlx-lm
    fi
    
    # Check for OpenMPI
    if ! conda list | grep -q "openmpi"; then
        echo "Installing OpenMPI..."
        conda install -c conda-forge openmpi -y
    else
        echo "✓ OpenMPI is installed"
    fi
else
    echo "ERROR: Conda environment '${CONDA_ENV}' not found!"
    echo "Creating new environment..."
    conda create -n "${CONDA_ENV}" python=3.12 -y
    conda activate "${CONDA_ENV}"
    conda install -c conda-forge openmpi -y
    pip install -U mlx-lm
fi

# Download pipeline_generate.py if needed
if [ ! -f "${MLX_PROJECT_PATH}/pipeline_generate.py" ]; then
    echo "Downloading pipeline_generate.py..."
    curl -O https://raw.githubusercontent.com/ml-explore/mlx-lm/refs/heads/main/mlx_lm/examples/pipeline_generate.py
else
    echo "✓ pipeline_generate.py exists"
fi

# Ensure hosts.json exists
if [ ! -f "${MLX_HOSTS_FILE}" ]; then
    echo "Creating hosts.json template..."
    cat <<JSON > "${MLX_HOSTS_FILE}"
[
  {"ssh": "$(hostname)"},
  {"ssh": "hostname2"},
  {"ssh": "hostname3"}
]
JSON
    echo "⚠ Please update ${MLX_HOSTS_FILE} with your actual hostnames!"
else
    echo "✓ hosts.json exists"
fi

echo
echo "==================================================="
echo "Setup verification complete!"
echo "Environment: ${CONDA_ENV}"
echo "==================================================="
