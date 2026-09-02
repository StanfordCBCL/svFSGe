# FSGe: A fast and strongly-coupled 3D fluid-solid-growth interaction method

## Reference
[arXiv:2404.14187](https://arxiv.org/abs/2404.14187)

## Quickstart

### One-Line Install (Recommended)

Install FSGe on a new machine with a single command:

```bash
curl -fsSL https://raw.githubusercontent.com/Eleven7825/svFSGe/master/scripts/install.sh | bash
```

This will clone the repository, set up the Docker environment, and drop you into an interactive shell ready to run simulations.

**Prerequisites**: Docker and git must be installed and Docker must be running.

### Manual Setup

1. Install docker and pull the svMultiphyiscs image from
```bash
docker pull simvascular/solver:latest
```
Make two directories, one for svMultiPhysics(svfsi), one for svFSGe:
```bash
mkdir svFSGe && mkdir svMultiPhysics
```

Start a new svMultiphysics container by:

```bash
docker run -it --user $(id -u):$(id -g) \
           -v ./svMultiPhysics:/svfsi \
           -v ./svFSGe:/svFSGe        \
              simvascular/solver:latest /bin/bash
```
2. Inside the container, install `svFSIplus` from [this branch](https://github.com/Eleven7825/svMultiPhysics/tree/FSGe), to build svfsi, you need to run(inside docker, command line begins with #):
```bash
cd /svfsi
git init
git remote add origin https://github.com/Eleven7825/svMultiPhysics.git
git fetch origin FSGe
git checkout -b FSGe origin/FSGe
```

At the /svfsi directory, build it with
```bash
bash makeCommand.sh
```

3. Install required Python packages (inside the container):
```bash
# Option 1: Install from requirements.txt (recommended)
pip install -r /svFSGe/requirements.txt

# Option 2: Install individually
pip install numpy vtk matplotlib scipy xmltodict distro
```

4. Still inside the container, you need to adapt paths in `/svFSGe/in_sim/partitioned_full.json` to make sure they are correct

5. Run the simulation:
```bash
cd /svFSGe
python3 ./fsg.py in_sim/partitioned_full.json
```

## Quick Setup Script

If you've already cloned the repository, use the setup script to create a persistent Docker environment:

```bash
# From the svFSGe directory
./scripts/setup_docker.sh
```

This script will:
- Create a persistent Docker container named `fsg-dev`
- Start the container with proper volume mounts
- Clone and build svFSIplus
- Install Python dependencies
- Provide an interactive shell ready to run simulations

The container persists after you exit. Run the script again to reconnect, or use:
```bash
docker exec -it fsg-dev /bin/bash
```

## HPC / Singularity Installation

For HPC environments where Docker is not available, use Singularity/Apptainer:

```bash
git clone https://github.com/Eleven7825/svFSGe.git ~/svFSGe
cd ~/svFSGe
./scripts/setup_singularity.sh
```

**Run test simulation:**
```bash
./scripts/run_simulation.sh in_sim/partitioned_test.json
```

**Or manually with Singularity:**
```bash
singularity exec \
  --bind ~/svfsi:/svfsi \
  --bind ~/svFSGe:/svFSGe \
  ~/svFSGe/singularity_images/simvascular-solver.sif \
  python3 /svFSGe/fsg.py /svFSGe/in_sim/partitioned_test.json
```

**Note**: Adjust the number of cores in your configuration files to match your HPC allocation.

**Prerequisites**: Singularity/Apptainer and git. See [README_SINGULARITY.md](README_SINGULARITY.md) for complete documentation.

## Continuous Integration

GitHub Actions automatically tests FSGe on every pull request and branch push:
- **Test configuration**: `in_sim/partitioned_test.json` (nmax=2 for faster testing)
- **Workflow**: `.github/workflows/test-fsg.yml`
- **Comparison**: Validates convergence metrics against reference baseline

See `test_reference/nmax_2/README.md` for details on reference data.


## File overview

- `cylinder.py` generates structured FSI hex-meshes with configuration files in `in_geo`
- `fsg.py` runs partitioned FSGe coupling using svFSIplus with
  - `in_sim` FSGe configuration files
  - `in_svfsi_plus` svFSIplus input files
  - `in_petsc` PETSc linear solver settings
- `post.py` generate line plots from FSGe results
- `svfsi.py` sets up, executes, and processes svFSIplus simulations
- `utilities.py` IQN-ILS filtering
- `vtk_functions.py` useful VTK functions for file IO
- `scripts/` utility scripts including:
  - `compare_results.py` - CI test comparison script
  - `setup_docker.sh` - Automated Docker environment setup
