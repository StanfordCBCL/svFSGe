#!/bin/bash
#
# FSGe Quick Installer
#
# One-line install:
#   curl -fsSL https://raw.githubusercontent.com/Eleven7825/svFSGe/master/scripts/install.sh | bash
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FSGe Quick Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check prerequisites
if ! command -v git &> /dev/null; then
    echo -e "${RED}ERROR: git is not installed${NC}"
    echo "Please install git first: https://git-scm.com/downloads"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Please start Docker first"
    exit 1
fi

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
echo -e "${GREEN}Starting Docker setup...${NC}"
echo ""
./scripts/setup_docker.sh

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Repository location: $INSTALL_DIR"
echo ""
echo "To reconnect to the container later:"
echo "  cd $INSTALL_DIR && ./scripts/setup_docker.sh"
echo ""
