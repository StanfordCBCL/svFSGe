#!/bin/bash
#
# Docker Environment Setup Script for FSGe
#
# This script automates the setup of the Docker environment for running FSGe simulations.
# It creates necessary directories, starts the Docker container, builds svFSIplus,
# and installs Python dependencies.
#
# Usage:
#   ./scripts/setup_docker.sh [--interactive] [--test]
#
# Options:
#   --interactive  Drop into interactive shell after setup (default)
#   --test         Set up test configuration (nmax=2) instead of full
#   --help         Show this help message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
INTERACTIVE=true
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --help)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print colored message
print_step() {
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_step "FSGe Docker Environment Setup"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project directory: $PROJECT_DIR"
echo ""

# Create necessary directories
print_step "Creating directories..."
mkdir -p "$PROJECT_DIR/../svMultiPhysics"
echo "✓ Created svMultiPhysics directory"

# Check if Docker image exists locally
print_step "Checking Docker image..."
if docker images simvascular/solver:latest | grep -q simvascular; then
    echo "✓ Docker image simvascular/solver:latest found"
else
    print_warning "Docker image not found locally. Pulling..."
    docker pull simvascular/solver:latest
fi

# Create Docker run script
print_step "Starting Docker container..."

if [ "$INTERACTIVE" = true ]; then
    echo "Starting interactive Docker session..."
    echo "This will set up the environment and drop you into a shell."
    echo ""

    docker run -it --rm \
        -v "$PROJECT_DIR/../svMultiPhysics:/svfsi" \
        -v "$PROJECT_DIR:/svFSGe" \
        simvascular/solver:latest \
        /bin/bash -c "
            set -e

            echo '================================='
            echo 'Setting up Git configuration...'
            echo '================================='
            git config --global --add safe.directory /svfsi
            git config --global --add safe.directory /svFSGe

            echo ''
            echo '================================='
            echo 'Cloning svMultiPhysics...'
            echo '================================='
            cd /svfsi
            if [ ! -d .git ]; then
                git init
                git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
                git fetch --depth=1 origin FSGe
                git checkout -b FSGe origin/FSGe
                echo '✓ svMultiPhysics cloned successfully'
            else
                echo '✓ svMultiPhysics already cloned'
            fi

            echo ''
            echo '================================='
            echo 'Building svFSIplus...'
            echo '================================='
            if [ ! -f svFSI-build/bin/svFSI ]; then
                bash makeCommand.sh
                echo '✓ svFSIplus built successfully'
            else
                echo '✓ svFSIplus already built'
            fi

            echo ''
            echo '================================='
            echo 'Installing Python dependencies...'
            echo '================================='
            pip install -q numpy vtk matplotlib scipy xmltodict distro
            echo '✓ Python packages installed'

            echo ''
            echo '================================='
            echo 'Setup Complete!'
            echo '================================='
            echo ''
            echo 'You are now in the Docker container.'
            echo ''
            echo 'To run a simulation:'
            if [ '$TEST_MODE' = true ]; then
                echo '  cd /svFSGe'
                echo '  python3 ./fsg.py in_sim/partitioned_test.json'
            else
                echo '  cd /svFSGe'
                echo '  python3 ./fsg.py in_sim/partitioned_full.json'
            fi
            echo ''
            echo 'To exit: type exit or press Ctrl+D'
            echo ''

            # Start interactive shell
            /bin/bash
        "
else
    # Non-interactive mode (for scripting)
    docker run --rm \
        -v "$PROJECT_DIR/../svMultiPhysics:/svfsi" \
        -v "$PROJECT_DIR:/svFSGe" \
        simvascular/solver:latest \
        /bin/bash -c "
            set -e
            git config --global --add safe.directory /svfsi
            git config --global --add safe.directory /svFSGe
            cd /svfsi
            if [ ! -d .git ]; then
                git init
                git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
                git fetch --depth=1 origin FSGe
                git checkout -b FSGe origin/FSGe
            fi
            if [ ! -f svFSI-build/bin/svFSI ]; then
                bash makeCommand.sh
            fi
            pip install -q numpy vtk matplotlib scipy xmltodict distro
            echo 'Setup complete'
        "
fi

echo ""
print_step "Done!"
