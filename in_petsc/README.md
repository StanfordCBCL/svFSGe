# PETSc linear solver configuration files
The tangential stiffness matrix of the CMMe material has the following properties:
- Stage 1, pre-loading: major and minor symmetries
- Stage 2, G&R: minor symmetries only

This can make Stage 2 challenging to solve since most GMRES+preconditioner setups are targeted towards symmetric problems. FSGe thus relies on the [MUMPS](https://mumps-solver.org/index.php) parallel direct solver, integrated via PETSc.