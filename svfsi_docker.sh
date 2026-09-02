#!/bin/bash
# Run svFSI inside the fsg-dev container where PETSc libraries are available.
# Translates the host working directory (/home/shiyi/svFSGe/...) to the
# container mount point (/svFSGe/...).
CWD=$(pwd | sed 's|/home/shiyi/svFSGe|/svFSGe|')
# OpenMPI leaks /dev/shm/sm_segment.* files when solver processes are killed/aborted.
# Over a long FSG run (hundreds of sequential solver calls) these accumulate and
# exhaust the container's small (64M) /dev/shm, making later solves fail with
# "not enough space for /dev/shm/sm_segment" → "disp simulation failed".
# Solvers run one at a time here, so any existing segments are stale: purge them
# before each call to keep /dev/shm clear.
docker exec fsg-dev bash -lc 'rm -f /dev/shm/sm_segment.* /dev/shm/vader_* 2>/dev/null' || true
docker exec -w "$CWD" fsg-dev /svfsi/svFSI-build/bin/svFSI "$@"
