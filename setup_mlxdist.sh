#!/usr/bin/env bash
set -e

# Paths and environment names
PROJECT_PATH=$PWD
CONDA_SCRIPT="/Users/alex/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="mlxdist"

#echo "1. Removing old project folder (if any) at $PROJECT_PATH ..."
#rm -rf "$PROJECT_PATH"

#echo "2. Re-creating project folder ..."
#mkdir -p "$PROJECT_PATH"
#cd "$PROJECT_PATH"

echo "3. Removing any existing conda environment named '$CONDA_ENV' ..."
conda env remove -n "$CONDA_ENV" -y || true

echo "4. Creating a new conda environment '$CONDA_ENV' with Python 3.12 ..."
conda create -n "$CONDA_ENV" python=3.12 -y

echo "5. Sourcing conda script (so 'conda activate' will work) ..."
if [ -f "$CONDA_SCRIPT" ]; then
  . "$CONDA_SCRIPT"
else
  echo "WARNING: Can't find conda activation script at $CONDA_SCRIPT"
  echo "         Ensure conda is installed or adjust the path."
fi

echo "6. Activating conda environment '$CONDA_ENV' ..."
conda activate "$CONDA_ENV"

echo "7. Installing Open MPI and mlx-lm ..."
conda install -c conda-forge openmpi -y
pip install -U mlx-lm

echo "8. Downloading pipeline_generate.py ..."
curl -O https://raw.githubusercontent.com/ml-explore/mlx-lm/refs/heads/main/mlx_lm/examples/pipeline_generate.py

#echo "9. Creating hosts.json ..."
#cat <<EOF > hosts.json
#[
#  {"ssh": "hostname1"},
#  {"ssh": "hostname2"}
#]
#EOF

echo
echo "==================================================="
echo "SETUP COMPLETE!"
echo "You can now run distributed jobs by using something like:"
echo
echo "mlx.launch \\"
echo "  --hostfile $PROJECT_PATH/hosts.json \\"
echo "  --backend mpi \\"
echo "  $PROJECT_PATH/pipeline_generate.py \\"
echo "  --prompt \"What number is larger 6.9 or 6.11?\" \\"
echo "  --max-tokens 128 \\"
echo "  --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
echo
echo "Make sure to also run 'conda activate $CONDA_ENV' in any new shell."
echo "==================================================="
