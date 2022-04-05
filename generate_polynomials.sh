#!/bin/bash
#SBATCH --partition=bluemoon
# Nodes: physical compute nodes. A compute node offers resources such as
# processors, volatile memory (RAM), permanent disk space (e.g. SSD),
# accelerators (e.g. GPU) etc. With --nodes, we request that a minimum (or
# maximum) number of nodes be allocated to this job. A core is the part of
# a processor that does the computations. A processor comprises multiple
# cores, as well as a memory controller, a bus controller, and possibly
# many other components. A CPU, according to Slurm, "consumable resource
# offered by a node and it refer to a socket, a core, or a hardware thread,
# based on the Slurm configuration."
#SBATCH --nodes=1
# Tasks: number of processors (CPUs)
# Launch a maximum number of tasks. Default is one task per node.
#SBATCH --ntasks=1
# #SBATCH --ntasks-per-node=8
# #SBATCH --cpus-per-task=4
# Mem: memory per processor (CPU); default is 14G
#SBATCH --mem=2G
# Time: max wall time (48 hour limit)
#SBATCH --time=0:15:00
#SBATCH --job-name=generate_poly
#SBATCH --output=out/%x_%j.out
#SBATCH --mail-user=ambertsc@uvm.edu
#SBATCH --mail-type=FAIL
cd ${SLURM_SUBMIT_DIR}
source activate tf2
time python generate_polynomials.py