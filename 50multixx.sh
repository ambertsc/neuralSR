#!/bin/sh

# Specify a partition
#SBATCH --partition=dggpu

# Request nodes
#SBATCH --nodes=1

# Request some processor cores
#SBATCH --ntasks=1

# GPU: number of GPUs
#SBATCH --gres=gpu:2

# Request memory
#SBATCH --mem=14G

# Run for 2 hours
#SBATCH --time=2:00:00

# Specify job array
#SBATCH --array=0-29

# Name job
#SBATCH --job-name=padding_

# Name output file (x=name, A=jobID, a=job index)
#SBATCH --output=out/%x_%j_%a.out

#SBATCH --mail-user=ambertsc@uvm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --verbose

# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

# Executable section: echoing some Slurm data
echo "Starting sbatch script myscript.sh at:`date`"
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"
echo "Task ID:	       ${SLURM_ARRAY_TASK_ID}"

source activate tf2
time python SGPT50xxmulti.py ${SLURM_ARRAY_TASK_ID}
