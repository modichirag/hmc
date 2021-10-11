#!/bin/bash
###SBATCH -N 2
##SBATCH -p ccm
##SBATCH -J funnel


wsize=50
posnum=0 
nsamples=1000

time mpirun -n $wsize python -u  test_posteriordb.py --posnumber=$posnum --nsamples=10000


##
##for ndim in 20 100; do
###for ndim in  2 5 10 20 100 ; do
##    for step in  5.0  2.0  1.0 0.5 0.2 0.1 0.05 0.01; do
##	for two_factor in 2 5 10; do
##	    echo $ndim $step $two_factor
##	    time mpirun -n $wsize python -u  funnelmpi_2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 100000   --lpath 5
##	    time mpirun -n $wsize python -u  funnelmpi_2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 100000   --lpath 10
##	    time mpirun -n $wsize python -u  funnelmpi_2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 100000   --lpath 20
##	    time mpirun -n $wsize python -u  funnelmpi_2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 100000   --lpath 30
##	done
##    done
##done
##
##
