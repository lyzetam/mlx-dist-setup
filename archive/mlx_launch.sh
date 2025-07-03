#!/bin/bash
# MLX Distributed Launcher

SCRIPT="${1}"
shift
ARGS="${@}"

# Number of processes (default 2)
NP="${NP:-2}"

# Hosts (default local)
HOSTS="${HOSTS:-localhost}"

echo "Launching MLX distributed..."
echo "Script: $SCRIPT"
echo "Processes: $NP"
echo "Hosts: $HOSTS"

if [ "$HOSTS" = "localhost" ]; then
    # Local run
    /opt/homebrew/bin/mpirun -np $NP /Users/zz/anaconda3/envs/mlx-distributed/bin/python mlx_dist_wrapper.py "$SCRIPT" $ARGS
else
    # Distributed run
    /opt/homebrew/bin/mpirun --host "$HOSTS" /Users/zz/anaconda3/envs/mlx-distributed/bin/python mlx_dist_wrapper.py "$SCRIPT" $ARGS
fi
