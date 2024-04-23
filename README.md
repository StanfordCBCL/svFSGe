# FSGe: A fast and strongly-coupled 3D fluid-solid-growth interaction method

## Reference
[arXiv:2404.14187](https://arxiv.org/abs/2404.14187)

## Quickstart

1. Install `svFSIplus` from [this branch](https://github.com/mrp089/svFSIplus/commit/e05e3f95b375329458b2e5f2d1f5ed5fb97df3d5) (including PETSc with MUMPS)
2. Adapt paths in `in_sim/partitioned_full.json`
3. Run
    ```bash
    ./fsg.py in_sim/partitioned_full.json
    ```
## File overview

- `cylinder.py` generates structured FSI hex-meshes with configuration files in `in_geo`
- `fsg.py` runs partiotioned FSGe coupling using svFSIplus with
  - `in_sim` FSGe configuration files
  - `in_svfsi_plus` svFSIplus input files
  - `in_petsc` PETSc linear solver settings
- `post.py` generate line plots from FSGe results
- `svfsi.py` sets up, executes, and processes svFSIplus simulations
- `utilities.py` IQN-ILS filtering
- `vtk_functions.py` useful VTK functions for file IO
- `scripts` more or less useful scripts
