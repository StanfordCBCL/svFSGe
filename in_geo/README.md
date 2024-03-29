# Input files for cylindrical hexahedral FSGe meshes
This folder contains mesh configuration files with different element sizes. All meshes contain a fluid and a solid mesh with a matching conforming interface and all the surfaces to apply boundary conditions in `svFSIplus`. The meshes are structured tetrahedral.

### Mesh parameters
- `f_out` name of output folder
- `r_inner` solid inner radius (= fluid radius)
- `r_outer` solid outer radius
- `height` vessel length
- `n_seg` number of cylinder segments: 1 (full, used), 4 (quarter, not used recently)
- `n_axi` number of axial elements
- `n_cir` number of circumferential elements
- `n_rad_gr` number of solid radial elements (CMMe can become unstable for `n_rad_gr>1` and `KsKi>0`)
- `n_rad_tran` number of transition elements in fluid "spider" mesh between quadratic core and circumferential elements
- `exp` exponent for axial mesh refinement in the middle
- `quad` convert to quadratic elements (not used, requires [additional code](https://github.com/vvedula22/useful_codes/tree/master/gambitToVTK) for conversion)