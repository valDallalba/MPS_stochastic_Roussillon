#! /bin/bash

#-*- coding: utf-8 -*-
#-----------------------------------------------------------------------------

#SBATCH -J 00_job_post_simulation.sh             # job name
#SBATCH -o 00_job_post_simulation.out         # output file, redirection of stdout (and stderr (merged))
#SBATCH -c 1               # number of cpu(s) per task (var: SLURM_CPUS_PER_TASK)


#SBATCH --mem-per-cpu=200GB     # required memory per CPU (in MB or specified unit)
                             #    (default:DefMemPerCPU / type: scontrol show config |grep DefMemPerCPU)
#SBATCH -t 24:00:00        # limit for total run time
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

time srun python3 run_post_simulation_mps.py 
echo -n "Normal terminaison"
