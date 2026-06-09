#!/bin/bash

#PBS -l nodes=1:ppn=64
#PBS -m abe
#PBS -N torus_512_64c

hostname

cd $PBS_O_WORKDIR

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo ------------------------------------------------------

mpirun -np 64 ./harm 8 8 1 2>err 1>out
