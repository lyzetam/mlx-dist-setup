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
