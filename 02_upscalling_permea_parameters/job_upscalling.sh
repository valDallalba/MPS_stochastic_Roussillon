#! /bin/bash

#-*- coding: utf-8 -*-
#-----------------------------------------------------------------------------

#SBATCH -J job_upscalling                # job name
#SBATCH -a 0-19%5                        # iteration name
#SBATCH -c 6                             # number of cpu(s) per task (var: SLURM_CPUS_PER_TASK)
#SBATCH -o job_upscalling_2022.out         # output file, redirection of stdout (and stderr (merged))
#SBATCH --partition=normal


#SBATCH --mem-per-cpu=20GB     # required memory per CPU (in MB or specified unit)
                              # (default:DefMemPerCPU / type: scontrol show config |grep DefMemPerCPU)

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

echo -n "Running :"
srun hostname
echo "Job_NAME =${SLURM_JOB_ID}, TASK = ${SLURM_ARRAY_TASK_ID}"
echo -n "Start at"
date

time srun python3 run_upscalling_2022.py -sim ${SLURM_ARRAY_TASK_ID}

echo -n "Normal terminaison"
echo -n "End at"
date
