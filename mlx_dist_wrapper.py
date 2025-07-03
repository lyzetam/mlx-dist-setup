#!/usr/bin/env python3
'''
MLX Distributed Wrapper
This ensures MLX properly connects to MPI processes
'''

import sys
import os

# CRITICAL: Set backend before importing MLX
os.environ['MLX_DISTRIBUTED_BACKEND'] = 'mpi'

# Import MPI first to ensure it's initialized
from mpi4py import MPI

# Now import MLX
import mlx.core as mx

# Get script to run
if len(sys.argv) < 2:
    print("Usage: mlx_dist_wrapper.py <script.py> [args...]")
    sys.exit(1)

script_path = sys.argv[1]
sys.argv = sys.argv[1:]  # Shift arguments for the target script

# Execute the script
with open(script_path, 'r') as f:
    code = f.read()
    exec(code, {'__name__': '__main__'})
