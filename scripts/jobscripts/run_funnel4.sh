#!/bin/bash
###SBATCH -N 2
#SBATCH -p ccm
#SBATCH -J funnel4

# Start from an "empty" module collection.
module purge
# Load in what we need to execute mpirun.
module load slurm gcc openmpi

## We assume this executable is in the directory from which you ran sbatch.
#mpirun ./mpi_hello_world

source activate defpyn

wsize=50
nsamples=100000



ndim=50
lpath=30


two_factor=2


# Set space as the delimiter
IFS=' '
#Read the split words into an array based on space delimiter

##for i in  "4 0.5" "4 0.2" "4 0.1" "3 0.2" "3 0.1" "2 0.1" "3 0.05" "2 0.05"; do
##    x=$i
##    read -a strarr <<< "$x"
##    nsub=${strarr[0]}
##    step=${strarr[1]}
##    echo $x
##    echo $nsub 
##    echo $step
##    echo $ndim $step $two_factor $nsub $lpath
##    time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
##    #time mpirun -n $wsize python -u  funnelmpi_multistep3.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
##done
##
##
##two_factor=10
##
##for i in  "2 0.5" "2 0.2" "2 0.1" "3 0.5" "3 0.2" "3 0.1" "2 0.05"; do
##    x=$i
##    read -a strarr <<< "$x"
##    nsub=${strarr[0]}
##    step=${strarr[1]}
##    echo $x
##    echo $nsub 
##    echo $step
##    echo $ndim $step $two_factor $nsub $lpath
##    time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
##    #time mpirun -n $wsize python -u  funnelmpi_multistep3.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
##done
##

two_factor=5

for i in  "2 0.2" "2 0.1" "3 0.5" "3 0.2" "4 0.5" "3 0.1" "2 0.05"; do
    x=$i
    read -a strarr <<< "$x"
    nsub=${strarr[0]}
    step=${strarr[1]}
    echo $x
    echo $nsub 
    echo $step
    echo $ndim $step $two_factor $nsub $lpath
    time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
    time mpirun -n $wsize python -u  funnelmpi_multistep3.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples $nsamples  --nsub $nsub --lpath $lpath
done
