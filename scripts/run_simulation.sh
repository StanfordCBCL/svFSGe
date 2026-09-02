#!/bin/bash
#
# Helper script to run FSGe simulations using Singularity
#
# Usage:
#   ./scripts/run_simulation.sh [config_file]
#
# Examples:
#   ./scripts/run_simulation.sh                          # Run default (partitioned_full.json)
#   ./scripts/run_simulation.sh in_sim/partitioned_test.json
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Singularity image path
SINGULARITY_IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

# Check if image exists
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo -e "${RED}ERROR: Singularity image not found${NC}"
    echo "Please run setup first: ./scripts/setup_singularity.sh"
    exit 1
fi

# Default config file
CONFIG_FILE="${1:-in_sim/partitioned_full.json}"

# Check if config file exists
if [ ! -f "$PROJECT_DIR/$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running FSGe Simulation${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Config: $CONFIG_FILE"
echo ""

# Run the simulation
singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$SINGULARITY_IMAGE" \
    python3 /svFSGe/fsg.py "/svFSGe/$CONFIG_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Simulation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
