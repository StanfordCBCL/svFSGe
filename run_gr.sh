#!/bin/bash

set -e

export DYLD_LIBRARY_PATH=$PETSC_DIR/lib:$DYLD_LIBRARY_PATH
export LDFLAGS="-Wl,-no_compact_unwind"

# compile source code
(cd ~/work/repos/svFSI_fork/build/svFSI-build && make)

# delete old mesh
rm -rf mesh_tube_fsi*

# generate new mesh
./cylinder.py in_geo/fsg_full_${1}.json

# delete old simulation output
rm -rf gr

# archive simulation
mkdir gr
cp ~/work/repos/svFSI_fork/Code/Source/svFSI/FEMbeCmm.cpp gr
cp in_svfsi/gr_full.inp gr

# run simulation
mpirun -np ${2} ~/work/repos/svFSI_fork/build/svFSI-build/bin/svFSI in_svfsi/gr_full.inp || true

# run post-processing
./post.py gr
