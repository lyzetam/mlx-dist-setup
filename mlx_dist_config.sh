#!/bin/bash
# MLX Distributed Configuration

# Use Homebrew MPI
export PATH="/opt/homebrew/bin:$PATH"
export MPICC=/opt/homebrew/bin/mpicc
export MPIRUN=/opt/homebrew/bin/mpirun

# Python from conda environment
export PYTHON=/Users/zz/anaconda3/envs/mlx-distributed/bin/python

# Function to run distributed MLX
run_mlx_dist() {
    /opt/homebrew/bin/mpirun "$@"
}

echo "MLX Distributed configured with:"
echo "  MPI: Homebrew OpenMPI 5.0.7"
echo "  Python: Conda environment (mlx-distributed)"
echo ""
echo "Usage: run_mlx_dist -np 4 python your_script.py"
