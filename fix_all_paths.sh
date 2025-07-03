#!/usr/bin/env bash

echo "==================================================="
echo "Fixing all hardcoded paths in MLX scripts"
echo "==================================================="
echo

# First, update env_config.sh with correct values
echo "Updating env_config.sh..."
cat > env_config.sh <<'EOF'
#!/usr/bin/env bash

# MLX Distributed Environment Configuration
# Updated for existing mlx-env setup

# User Configuration
export MLX_USER="zz"

# Path Configuration
export MLX_PROJECT_PATH="/Users/zz/Documents/GitHub/mlx-dist-setup"
export CONDA_BASE_PATH="/Users/zz/anaconda3"

# Conda Configuration - Using existing mlx-env
export CONDA_ENV="mlx-env"
export CONDA_SETUP="${CONDA_BASE_PATH}/etc/profile.d/conda.sh"

# Network Configuration (update this with your actual network interface)
export MLX_NETWORK_INTERFACE="en0"  # Changed from lo0

# Memory Configuration (for Apple Silicon)
export MLX_MEMORY_LIMIT_MB="10000"

# Model Configuration
export MLX_MODEL="mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
export MLX_DEFAULT_PROMPT="What number is larger 6.9 or 6.11?"
export MLX_MAX_TOKENS="128"

# Host Configuration File
export MLX_HOSTS_FILE="${MLX_PROJECT_PATH}/hosts.json"

# Python Path
export MLX_PYTHON_PATH="${CONDA_BASE_PATH}/envs/${CONDA_ENV}/bin/python"

# Verify critical paths exist
check_paths() {
    local missing=0
    
    if [ ! -d "${CONDA_BASE_PATH}" ]; then
        echo "ERROR: Conda base path not found: ${CONDA_BASE_PATH}"
        echo "       Please update CONDA_BASE_PATH in env_config.sh"
        missing=1
    fi
    
    if [ ! -f "${CONDA_SETUP}" ]; then
        echo "ERROR: Conda setup script not found: ${CONDA_SETUP}"
        echo "       Please update CONDA_BASE_PATH in env_config.sh"
        missing=1
    fi
    
    return $missing
}

# Function to create project directory if it doesn't exist
ensure_project_dir() {
    if [ ! -d "${MLX_PROJECT_PATH}" ]; then
        echo "Creating project directory: ${MLX_PROJECT_PATH}"
        mkdir -p "${MLX_PROJECT_PATH}"
    fi
}
EOF

# Now fix run_mlx.sh
echo "Fixing run_mlx.sh..."
cat > run_mlx.sh <<'EOF'
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
EOF

# Fix setup_mlxdist.sh
echo "Fixing setup_mlxdist.sh..."
cat > setup_mlxdist.sh <<'EOF'
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
EOF

# Fix set_mem_limit.sh
echo "Fixing set_mem_limit.sh..."
cat > set_mem_limit.sh <<'EOF'
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
EOF

# Make all scripts executable
chmod +x env_config.sh
chmod +x run_mlx.sh
chmod +x setup_mlxdist.sh
chmod +x set_mem_limit.sh
chmod +x copy_setup_script.sh 2>/dev/null || true

echo
echo "✓ All scripts have been fixed!"
echo
echo "Summary of changes:"
echo "- Removed all hardcoded paths to 'alex'"
echo "- Updated conda environment to 'mlx-env'"
echo "- Changed network interface from lo0 to en0"
echo "- All scripts now use env_config.sh for configuration"
echo
echo "Next steps:"
echo "1. Update hosts.json with your actual machine hostnames"
echo "2. Run: ./copy_setup_script.sh to copy to remote machines"
echo "3. Run: ./run_mlx.sh to start distributed job"
echo
echo "==================================================="