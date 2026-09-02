#!/bin/bash
#
# Singularity Environment Setup Script for FSGe (HPC Version)
#
# This script automates the setup of the Singularity environment for running FSGe simulations.
# It converts Docker image to Singularity, builds svFSIplus, and installs Python dependencies.
#
# Usage:
#   ./scripts/setup_singularity.sh [--shell] [--test] [--help]
#
# Options:
#   --shell        Drop into interactive shell after setup
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
INTERACTIVE_SHELL=false
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --shell)
            INTERACTIVE_SHELL=true
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

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    print_error "Singularity/Apptainer is not installed. Please contact your HPC administrator."
    exit 1
fi

print_step "FSGe Singularity Environment Setup"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project directory: $PROJECT_DIR"
echo "Singularity version: $(singularity --version)"
echo ""

# Create necessary directories
print_step "Creating directories..."
mkdir -p "$PROJECT_DIR/../svfsi"
mkdir -p "$PROJECT_DIR/singularity_images"
echo "✓ Created necessary directories"

# Singularity image path
SINGULARITY_IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

# Check if Singularity image exists
print_step "Checking Singularity image..."
if [ -f "$SINGULARITY_IMAGE" ]; then
    echo "✓ Singularity image found: $SINGULARITY_IMAGE"
else
    print_warning "Singularity image not found. Building from Docker image..."
    echo "This may take several minutes..."

    # Pull Docker image and convert to Singularity
    singularity build "$SINGULARITY_IMAGE" docker://simvascular/solver:latest

    if [ $? -eq 0 ]; then
        echo "✓ Singularity image created successfully"
    else
        print_error "Failed to create Singularity image"
        exit 1
    fi
fi

# Check if svFSIplus is already built
SVFSI_BUILT=false
if [ -f "$PROJECT_DIR/../svfsi/svFSI-build/bin/svFSI" ]; then
    echo ""
    echo -e "${GREEN}✓ svFSIplus already built${NC}"
    SVFSI_BUILT=true
fi

# Check if Python packages are installed (by checking for a marker file)
PYTHON_INSTALLED=false
if [ -f "$PROJECT_DIR/.python_deps_installed" ]; then
    echo -e "${GREEN}✓ Python dependencies already installed${NC}"
    PYTHON_INSTALLED=true
fi

# Run setup if needed
if [ "$SVFSI_BUILT" = false ] || [ "$PYTHON_INSTALLED" = false ]; then
    print_step "Running initial setup..."

    # Create a setup script to run inside Singularity
    cat > /tmp/svfsge_setup.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "================================="
echo "Setting up Git configuration..."
echo "================================="
git config --global --add safe.directory /svfsi
git config --global --add safe.directory /svFSGe

echo ""
echo "================================="
echo "Cloning svMultiPhysics..."
echo "================================="
cd /svfsi
if [ ! -d .git ]; then
    git init
    git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
    git fetch --depth=1 origin FSGe
    git checkout -b FSGe origin/FSGe
    echo "✓ svMultiPhysics cloned successfully"
else
    echo "✓ svMultiPhysics already cloned"
fi

echo ""
echo "================================="
echo "Building svFSIplus..."
echo "================================="
if [ ! -f svFSI-build/bin/svFSI ]; then
    bash makeCommand.sh
    echo "✓ svFSIplus built successfully"
else
    echo "✓ svFSIplus already built"
fi

echo ""
echo "================================="
echo "Installing Python dependencies..."
echo "================================="
echo "This may take a few minutes (packages: numpy, vtk, matplotlib, scipy, xmltodict, distro, meshio)"
echo ""

# Install to user directory explicitly (works with read-only container)
pip install --user --no-warn-script-location \
    numpy>=1.20.0 \
    vtk>=9.0.0 \
    matplotlib>=3.3.0 \
    scipy>=1.6.0 \
    xmltodict>=0.12.0 \
    distro>=1.5.0 \
    meshio>=5.0.0

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Python packages installed successfully"

    # Verify packages can be imported
    echo ""
    echo "Verifying Python packages..."
    python3 -c "import numpy, vtk, matplotlib, scipy, xmltodict, distro, meshio; print('✓ All packages verified and importable')"

    if [ $? -eq 0 ]; then
        # Create marker file only if verification succeeds
        touch /svFSGe/.python_deps_installed
    else
        echo "✗ Package verification failed"
        exit 1
    fi
else
    echo ""
    echo "✗ Python package installation failed"
    exit 1
fi

echo ""
echo "================================="
echo "Setup Complete!"
echo "================================="
SETUP_SCRIPT

    chmod +x /tmp/svfsge_setup.sh

    # Run setup inside Singularity container
    # Note: No --writable-tmpfs needed since pip installs to user home directory
    singularity exec \
        --bind "$PROJECT_DIR/../svfsi:/svfsi" \
        --bind "$PROJECT_DIR:/svFSGe" \
        "$SINGULARITY_IMAGE" \
        /bin/bash /tmp/svfsge_setup.sh

    rm /tmp/svfsge_setup.sh

    echo ""
    echo -e "${GREEN}✓ Initial setup complete${NC}"
fi

echo ""
print_step "Environment ready!"
echo ""
echo "To run a simulation:"
if [ "$TEST_MODE" = true ]; then
    echo "  singularity exec --bind \"$PROJECT_DIR/../svfsi:/svfsi\" --bind \"$PROJECT_DIR:/svFSGe\" \\"
    echo "    $SINGULARITY_IMAGE \\"
    echo "    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_test.json"
else
    echo "  singularity exec --bind \"$PROJECT_DIR/../svfsi:/svfsi\" --bind \"$PROJECT_DIR:/svFSGe\" \\"
    echo "    $SINGULARITY_IMAGE \\"
    echo "    python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_full.json"
fi
echo ""
echo "Or use the helper script:"
echo "  $PROJECT_DIR/scripts/run_simulation.sh"
echo ""

# Drop into interactive shell if requested
if [ "$INTERACTIVE_SHELL" = true ]; then
    print_step "Entering interactive shell..."
    echo "You are now in the Singularity container."
    echo "Working directory: /svFSGe"
    echo "To exit: type 'exit' or press Ctrl+D"
    echo ""

    singularity shell \
        --bind "$PROJECT_DIR/../svfsi:/svfsi" \
        --bind "$PROJECT_DIR:/svFSGe" \
        --pwd /svFSGe \
        "$SINGULARITY_IMAGE"
fi

echo ""
print_step "Done!"
echo ""
echo "Quick commands:"
echo "  Interactive shell:  ./scripts/setup_singularity.sh --shell"
echo "  Run simulation:     ./scripts/run_simulation.sh"
echo ""
