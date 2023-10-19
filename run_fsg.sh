#!/bin/bash

export LDFLAGS="-Wl,-no_compact_unwind"
(cd ~/work/repos/svFSI_fork/build/svFSI-build && make)
./fsg.py in_sim/partitioned_full.json
