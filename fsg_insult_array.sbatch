#!/bin/bash
#SBATCH --job-name=fsg_insult
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16G
#SBATCH --array=0-23
#SBATCH --output=array_runs/slurm-%A_%a.out

# Diverse-insult FSG array job (real CFD). Each task runs one trajectory with a
# distinct spatial G&R insult (in_sim/insult_array/traj_NN.json). Every task runs
# in its own working directory so the per-run scratch (mesh_tube_fsi, trajNN_*
# output) never collides between concurrent array tasks.

PROJECT_DIR="$HOME/svFSGe"
IMAGE="$PROJECT_DIR/singularity_images/simvascular-solver.sif"

TID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
CFG="in_sim/insult_array/traj_${TID}.json"
RUNDIR="$PROJECT_DIR/array_runs/traj_${TID}"

mkdir -p "$RUNDIR"

echo "[$(date)] task $TID  host=$(hostname)  cfg=$CFG  rundir=$RUNDIR"

singularity exec \
    --bind "$PROJECT_DIR/../svfsi:/svfsi" \
    --bind "$PROJECT_DIR:/svFSGe" \
    --pwd "/svFSGe/array_runs/traj_${TID}" \
    "$IMAGE" \
    python3 /svFSGe/fsg.py "/svFSGe/${CFG}"

echo "[$(date)] task $TID done rc=$?"
