#!/bin/sh
set -eu

PROGRAM="$1"
HOST_LIST="$2"
NUM_NODES="$3"

MPIRUN="/opt/openmpi-1.10.2/bin/mpirun"
#MPIRUN_FLAGS="-mca btl tcp,self"
MPIRUN_FLAGS="-mca btl tcp,self -x LD_LIBRARY_PATH=/opt/openmpi-1.10.2/lib:/nscratch/phj/local/cudnn_v4/lib64 -x RUST_BACKTRACE=1 -x RUST_LOG=info"

${MPIRUN} ${MPIRUN_FLAGS} -machinefile ${HOST_LIST} -n ${NUM_NODES} -npernode 1 ${PROGRAM}
