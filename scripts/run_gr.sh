#!/bin/bash

set -e

export DYLD_LIBRARY_PATH=$PETSC_DIR/lib:$DYLD_LIBRARY_PATH
export LDFLAGS="-Wl,-no_compact_unwind"
export SVFSI_DIR="/Users/pfaller/work/repos/svFSIplus_fork"

# compile source code
(cd $SVFSI_DIR/build/svFSI-build && make -j8)

# delete old mesh
rm -rf mesh_tube_fsi*

# generate new mesh
./cylinder.py in_geo/fsg_full_${1}.json

# delete old simulation output
rm -rf gr

# archive simulation
mkdir gr
cp $SVFSI_DIR/Code/Source/svFSI/gr_equilibrated.cpp gr
cp in_svfsi_plus/gr_full.xml gr
cp in_geo/fsg_full_${1}.json gr

# run simulation
mpirun -np ${2} $SVFSI_DIR/build/svFSI-build/bin/svFSI in_svfsi_plus/gr_full.xml || true

# run post-processing
./post.py gr
