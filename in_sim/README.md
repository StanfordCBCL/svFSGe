# Configuration files for FSGe
Contains an example for an FSGe configuration file
### Parameters of configuration file
- `fsi` true (run fluid simulation), false (approximate Poiseuille solution)
- `debug` if true, prints `svFSIplus` simulation output to screen
- `mesh` geometry input file from `in_geo`
- `n_procs` number of processors to use for `fluid`, `mesh`, and `solid` simulations
- `fluid` parameters for Poiseuille flow solution
- `n_max` number of time steps in `fluid`, `mesh`, and `solid` simulations
- `coup` coupling parameters
  - `nmax` maximum number of coupling iterations
  - `nmin` minimum number of coupling iterations
  - `tol` coupling tolerance for convergence
  - `method` coupling method: `iqn_ils` (recommended), `static`, `aitken` (slow, likely unstable)
  - `omega0` damping parameter for static relaxation
  - `iqn_ils_q` IQN-ILS number of old iterations to use
  - `iqn_ils_eps` IQN-ILS filtering tolerance
- `nmax` number of G&R load steps (plus one pre-loading step)
- `exe` executable file paths for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `inp` input files for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `interfaces` names for various surfaces and input files for `fluid`, `mesh`, and `solid` simulations (should not be changed)
- `out` name of output folders for `fluid`, `mesh`, and `solid` simulations (must match with `svFSIplus` input files)
- `name` name of the FSGe simulation
- `paths_*` folder names for various operating systems