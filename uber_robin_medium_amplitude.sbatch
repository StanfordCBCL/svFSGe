#!/bin/bash
#SBATCH --job-name=uber_robin_amp
#SBATCH --partition=week
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G

# Mirrors uber_robin_medium_magnitude.sbatch (uber_robin coupling, no
# neural-operator surrogate, medium-resolution mesh, PULSATILE flow, 2
# cardiac cycles x 200 steps) but with WSS pulsatile field reduction set to
# "amplitude" (true max_k(||WSS_k(p)||) - min_k(||WSS_k(p)||), peak-to-peak
# of the magnitude signal) instead of "magnitude" (mean of magnitude).
# Only the config differs.
#
# No srun: srun/interactive jobs are QOS-capped at 4 concurrent per user
# and not even allowed on the week partition, but sbatch jobs get "normal"
# QOS regardless. This job just runs its own command directly.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd /svFSGe \
    "$IMAGE" \
    python3 /svFSGe/fsg.py /svFSGe/in_sim/uber_robin_medium_pulsatile_amplitude.json
