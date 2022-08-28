#!/bin/bash
###SBATCH -N 2
##SBATCH -p ccm
##SBATCH -J funnel

# Start from an "empty" module collection.
#module purge
# Load in what we need to execute mpirun.
#module load slurm gcc openmpi

#source activate defpyn



wsize=50



for lpath in 10 ; do #20 10; do
    for ndim in 100 ; do
	for step in  0.04 0.02 ; do
	    echo $ndim $step
	    time mpirun -n $wsize python -u  funnelmpi.py --ndim $ndim --step_size $step --nsamples 100000  --lpath $lpath
	done
    done
done

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
