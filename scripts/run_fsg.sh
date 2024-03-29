#!/bin/bash

# petsc
export PETSC_ARCH=arch-darwin-c-opt
export PETSC_DIR=/Users/pfaller/work/repos/petsc/$PETSC_ARCH
export PATH=$PETSC_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PETSC_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PETSC_DIR/lib:$LIBRARY_PATH
export DYLD_LIBRARY_PATH=$PETSC_DIR/lib:$DYLD_LIBRARY_PATH

export LDFLAGS="-Wl,-no_compact_unwind"
(cd ~/work/repos/svFSIplus_fork/build/svFSI-build && make)
./fsg.py in_sim/partitioned_full.json
