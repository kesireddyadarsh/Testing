#!/bin/bash
#SBATCH -n 1

mkdir $SLURM_JOBID
cp simulation_test_2 ./$SLURM_JOBID

cd $SLURM_JOBID
srun ./simulation_test_2
cd ..
