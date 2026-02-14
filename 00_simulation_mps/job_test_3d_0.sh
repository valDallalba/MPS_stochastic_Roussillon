#! /bin/bash

#-*- coding: utf-8 -*-
#-----------------------------------------------------------------------------

#SBATCH -J job_test_3d_0.sh             # job name
#SBATCH -o job_test_3d_0.out         # output file, redirection of stdout (and stderr (merged))
#SBATCH -c 30               # number of cpu(s) per task (var: SLURM_CPUS_PER_TASK)
#SBATCH --partition=normal

#SBATCH --mem-per-cpu=4GB     # required memory per CPU (in MB or specified unit)
                             #    (default:DefMemPerCPU / type: scontrol show config |grep DefMemPerCPU)
#SBATCH -t 120:00:00        # limit for total run time
                             #    format: "minutes", "minutes:seconds", "hours:minutes:seconds",
                             #    "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
                             #    (default: partition DEFAULTTIME / type: sinfo -o "%P %L")

#-----------------------------------------------------------------------------
# * sbatch submits a batch script to Slurm (and exit).
# * srun run a parallel job on cluster managed by Slurm.
#      - srun command in a batch script submitted by sbatch define a job step and inherits pertinent options given by #SBATCH.
#      - Option --exclusive passed to srun ensure that distinct CPU(s) are used for each job step.
#=============================================================================

echo -n "Running on:"
srun hostname

module load rlm_unine-lic

export OMP_NUM_THREADS=30
time srun python3 simu_mps_2021_int0.py 
echo -n "Normal terminaison"
