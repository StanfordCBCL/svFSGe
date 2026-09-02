#!/bin/bash
#
# FSGe Quick Installer for HPC/Singularity
#
# Adapted from the Docker version for HPC environments using Singularity/Apptainer
#
# Usage:
#   bash scripts/install_singularity.sh [install_dir]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FSGe Quick Installer (Singularity/HPC)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check prerequisites
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git is not installed${NC}"
    echo "Please contact your HPC administrator to install git"
    exit 1
fi

if ! command -v singularity &> /dev/null; then
    echo -e "${RED}ERROR: Singularity/Apptainer is not installed${NC}"
    echo "Please contact your HPC administrator"
    exit 1
fi

echo -e "${GREEN}✓ Found Singularity/Apptainer: $(singularity --version)${NC}"
echo ""

# Determine installation directory
INSTALL_DIR="${1:-$HOME/svFSGe}"

if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Directory $INSTALL_DIR already exists${NC}"
    read -p "Remove and reinstall? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$INSTALL_DIR"
    else
        echo "Installation cancelled"
        exit 1
    fi
fi

echo "Installing to: $INSTALL_DIR"
echo ""

# Clone repository
echo -e "${GREEN}Cloning repository...${NC}"
git clone https://github.com/Eleven7825/svFSGe.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo ""
echo -e "${GREEN}✓ Repository cloned${NC}"
echo ""

# Run setup script
echo -e "${GREEN}Starting Singularity setup...${NC}"
echo ""
./scripts/setup_singularity.sh

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Repository location: $INSTALL_DIR"
echo ""
echo "To run simulations later:"
echo "  cd $INSTALL_DIR && ./scripts/setup_singularity.sh --shell"
echo ""
