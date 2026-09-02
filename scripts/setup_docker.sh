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
mkdir -p "$PROJECT_DIR/../svfsi"
echo "✓ Created svfsi directory"

# Check if Docker image exists locally
print_step "Checking Docker image..."
if docker images simvascular/solver:latest | grep -q simvascular; then
    echo "✓ Docker image simvascular/solver:latest found"
else
    print_warning "Docker image not found locally. Pulling..."
    docker pull simvascular/solver:latest
fi

# Container name for persistence
CONTAINER_NAME="fsg-dev"

# Check if container already exists
if docker ps -a -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
    print_step "Found existing container: $CONTAINER_NAME"

    # Check if container is running
    if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
        echo "✓ Container is already running"
    else
        echo "Starting stopped container..."
        docker start $CONTAINER_NAME
        echo "✓ Container started"
    fi

    echo ""
    echo "Entering container shell..."
    echo "To exit: type 'exit' or press Ctrl+D"
    echo ""

    # Exec into existing container
    TTY_FLAG=$([ -t 0 ] && echo "-it" || echo "-i")
    docker exec $TTY_FLAG $CONTAINER_NAME /bin/bash

    echo ""
    print_step "Done!"
    exit 0
fi

# Create Docker run script
print_step "Creating new persistent container: $CONTAINER_NAME"

if [ "$INTERACTIVE" = true ]; then
    echo "This will set up the environment and drop you into a shell."
    echo "Container will persist after exit - rerun this script to reconnect."
    echo ""

    # Start container in detached mode with infinite sleep
    docker run -d --name $CONTAINER_NAME \
        -v "$PROJECT_DIR/../svfsi:/svfsi" \
        -v "$PROJECT_DIR:/svFSGe" \
        simvascular/solver:latest \
        sleep infinity

    echo "✓ Container created"
    echo ""

    # Run setup commands inside the container
    print_step "Running initial setup inside container..."

    docker exec $CONTAINER_NAME /bin/bash -c "
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
            pip install -q numpy>=1.20.0 vtk>=9.0.0 matplotlib>=3.3.0 scipy>=1.6.0 xmltodict>=0.12.0 distro>=1.5.0 meshio>=5.0.0
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
        "

    echo "✓ Initial setup complete"
    echo ""

    # Drop into interactive shell
    print_step "Entering container shell..."
    echo "To exit: type 'exit' or press Ctrl+D"
    echo "Tip: The container persists. Rerun this script to reconnect."
    echo ""

    TTY_FLAG=$([ -t 0 ] && echo "-it" || echo "-i")
    docker exec $TTY_FLAG $CONTAINER_NAME /bin/bash

else
    # Non-interactive mode (for scripting)
    echo "Starting container in background..."

    docker run -d --name $CONTAINER_NAME \
        -v "$PROJECT_DIR/../svfsi:/svfsi" \
        -v "$PROJECT_DIR:/svFSGe" \
        simvascular/solver:latest \
        sleep infinity

    echo "✓ Container created"
    echo ""
    print_step "Running setup in background..."

    docker exec $CONTAINER_NAME /bin/bash -c "
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
        pip install -q numpy>=1.20.0 vtk>=9.0.0 matplotlib>=3.3.0 scipy>=1.6.0 xmltodict>=0.12.0 distro>=1.5.0 meshio>=5.0.0
        echo 'Setup complete'
    "

    echo "✓ Setup complete"
    echo ""
    echo "Container '$CONTAINER_NAME' is running in the background."
    echo "To access it: docker exec -it $CONTAINER_NAME /bin/bash"
    echo "To stop it: docker stop $CONTAINER_NAME"
    echo "To remove it: docker rm -f $CONTAINER_NAME"
fi

echo ""
print_step "Done!"
echo ""
echo "Container management commands:"
echo "  Connect:  ./scripts/setup_docker.sh  (or: docker exec -it $CONTAINER_NAME /bin/bash)"
echo "  Stop:     docker stop $CONTAINER_NAME"
echo "  Start:    docker start $CONTAINER_NAME"
echo "  Remove:   docker rm -f $CONTAINER_NAME"
echo ""
