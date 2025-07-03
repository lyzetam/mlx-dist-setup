#!/bin/bash
# Install Miniconda on remote Mac nodes

echo "=== Installing Miniconda on Remote Nodes ==="
echo

REMOTE_HOSTS=("mm@mm1.local" "mm@mm2.local")

for host in "${REMOTE_HOSTS[@]}"; do
    echo "Installing on $host..."
    
    ssh $host 'bash -s' << 'INSTALL_SCRIPT'
#!/bin/bash
set -e

echo "  Checking if conda is already installed..."
if command -v conda &> /dev/null; then
    echo "  ✓ Conda already installed at $(which conda)"
    exit 0
fi

echo "  Installing Miniconda..."

# Download Miniconda for Apple Silicon
cd ~/
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install silently
bash Miniconda3-latest-MacOSX-arm64.sh -b -p ~/miniconda3

# Clean up installer
rm Miniconda3-latest-MacOSX-arm64.sh

# Initialize conda for zsh (default on macOS)
~/miniconda3/bin/conda init zsh
~/miniconda3/bin/conda init bash

echo "  ✓ Miniconda installed successfully"
echo "  Note: You may need to restart your shell or run 'source ~/.zshrc'"

# Also install Homebrew if not present (needed for OpenMPI)
if ! command -v brew &> /dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/null
fi

# Install OpenMPI via Homebrew
if command -v brew &> /dev/null; then
    echo "  Installing OpenMPI..."
    brew install open-mpi
fi

INSTALL_SCRIPT

    echo "✓ Completed setup on $host"
    echo
done

echo "=== Miniconda Installation Complete ==="
echo "Next: Run the deployment script again"