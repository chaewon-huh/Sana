#!/usr/bin/env bash
set -e

# Check if conda is already installed
if command -v conda &> /dev/null
then
    echo "Conda is already installed."
    exit 0
fi

echo "Conda not found. Installing Miniconda..."

# Set Miniconda installation directory
MINICONDA_DIR="$HOME/miniconda3"

# Determine OS and architecture
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)

case "$OS_TYPE" in
    Linux)
        case "$ARCH_TYPE" in
            x86_64)
                MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                ;;
            aarch64)
                MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                ;;
            *)
                echo "Unsupported architecture: $ARCH_TYPE"
                exit 1
                ;;
        esac
        ;;
    Darwin) # macOS
        case "$ARCH_TYPE" in
            x86_64)
                MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
                ;;
            arm64)
                MINICONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
                ;;
            *)
                echo "Unsupported architecture: $ARCH_TYPE"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unsupported operating system: $OS_TYPE"
        exit 1
        ;;
esac

MINICONDA_INSTALLER_SCRIPT="Miniconda3-latest.sh"

# Download Miniconda installer
echo "Downloading Miniconda installer from $MINICONDA_INSTALLER_URL..."
curl -L $MINICONDA_INSTALLER_URL -o $MINICONDA_INSTALLER_SCRIPT

# Install Miniconda
echo "Installing Miniconda to $MINICONDA_DIR..."
bash $MINICONDA_INSTALLER_SCRIPT -b -p $MINICONDA_DIR

# Clean up installer script
rm $MINICONDA_INSTALLER_SCRIPT

# Initialize conda for the current shell
echo "Initializing Conda..."
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"

# Optionally, add conda initialization to your shell profile (e.g., .bashrc or .zshrc)
# You might need to restart your shell or source the profile for changes to take effect.
# conda init bash # Or conda init zsh

echo "Attempting to initialize Conda for future bash sessions..."
$MINICONDA_DIR/bin/conda init bash

echo "Miniconda installation complete."
echo "You may need to restart your shell or run 'source ~/.bashrc' (or ~/.zshrc) for conda command to be available." 