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
- `nloads` number of G&R load steps (plus one pre-loading step). *(Renamed from
  `nmax`, which clashed with `coup.nmax`; the old name is still accepted.)*
- `gr_load` *(optional)* shape of how the G&R insult is ramped over the load steps.
  Previously this profile was hard-coded inside `gr_equilibrated.cpp`; it is now
  injected into the solid input file so it can be changed without rebuilding `svFSIplus`.
  Omit this section to keep the historical default (`tanh`, `steep` = 2.0).
  - `profile` ramp shape: `linear`, `tanh` (front-loaded, default), `power`, or `file`
  - `steep` shape parameter — `tanh` steepness or `power` exponent (default 2.0)
  - `curve` only for `profile: "file"` — the load factor **per step**, as a list of
    `[step, factor]` pairs (or a plain list of factors). `step` is the integer load-step
    number: `0` = pre-stress, `1..nloads` = G&R loads, so the curve has `nloads + 1`
    entries — one per step, no interpolation between steps. Example for `nloads = 3`:
    `[[0, 0.0], [1, 0.33], [2, 0.67], [3, 1.0]]`
- `gr_insult` *(optional)* **spatial** shape of the aneurysm insult (the elastin/stimulus
  knock-down that localizes the aneurysm), formerly hard-coded in `gr_equilibrated.cpp`.
  The knock-down at a point is `mag · f_axi(z) · f_cir(azimuth)`. Omit this section to keep
  the historical super-Gaussian default.
  - `profile` axial shape: `gaussian` (default) or `file`
  - `mag` peak elastin loss fraction (default 0.7)
  - `z_loc`, `z_wid` axial center / width as a fraction of the tube length (default 0.5, 0.25)
  - `z_exp` axial super-Gaussian exponent (default 2)
  - `asym` apply azimuthal (circumferential) localization (default `true`)
  - `theta_wid`, `theta_exp` azimuthal width (fraction of π) / exponent (default 0.55, 6)
  - `curve` only for `profile: "file"` — the **axial** factor `f_axi` as a list of
    `[z/lo, factor]` pairs (any function); `z/lo` is the normalized axial position in
    `[0, 1]`, linearly interpolated. The azimuthal factor still follows `asym`/`theta_*`.
    Example: `[[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]]`
- `exe` executable file paths for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `inp` input files for `svFSIplus` for `fluid`, `mesh`, and `solid` simulations
- `interfaces` names for various surfaces and input files for `fluid`, `mesh`, and `solid` simulations (should not be changed)
- `out` name of output folders for `fluid`, `mesh`, and `solid` simulations (must match with `svFSIplus` input files)
- `name` name of the FSGe simulation
- `paths_*` folder names for various operating systems