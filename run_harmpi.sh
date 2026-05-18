#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -m abe
#PBS -N harmpi_Michelle_test

hostname

cd $PBS_O_WORKDIR

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo ------------------------------------------------------

mpirun -np 1 ./harm 1 1 1 2>err 1>out
