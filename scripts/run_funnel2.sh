#!/bin/bash
wsize=30



for ndim in 10 20; do
#for ndim in 2 5 10 20; do
    for step in  0.2 0.1 0.05 0.01 0.005; do
	echo $ndim $step
	#time mpirun -n $wsize python -u  funnelstan.py --ndim $ndim --step_size $step --nsamples 5000  --lpath 10
    done
done


#for ndim in 2 5 10 20 ; do
for ndim in 5; do
#    for step in  1.0 0.5 0.2 0.1 0.05 0.01 0.005; do
    for step in  0.5  0.1 0.05 0.01; do
	echo $ndim $step
	time mpirun -n $wsize python -u  funnelmpi.py --ndim $ndim --step_size $step --nsamples 10000  --lpath 10
	time mpirun -n $wsize python -u  funnelmpi.py --ndim $ndim --step_size $step --nsamples 10000  --lpath 5
    done
done


#for ndim in 2 5 10 20; do
for ndim in  2 5 10 20 ; do
    for step in  1.0 0.5 0.2 0.1 0.05 0.01; do
	for two_factor in 2 5 10; do
	    echo $ndim $step $two_factor
	    #time mpirun -n $wsize python -u  funnelmpi_2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000   --lpath 10
	done
    done
done


for ndim in 2 5 10 20; do
    for step in 2.0 1.0 0.5 0.2 0.1 ; do
	for two_factor in 2 ; do
	    for nsub in 2 3 4 5 6; do
		echo $ndim $step $two_factor $nsub
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 5
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 10
	    done
	done
    done
done

for ndim in  2 5 10 20; do
    for step in  2.0  1.0 0.5 0.2  ; do
	for two_factor in 5 ; do
	    for nsub in 2 3 4 ; do
		echo $ndim $step $two_factor $nsub
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 5
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 10
	    done 
	done
    done
done


for nsub in 2 3; do 
    for ndim in  2 5 10 20; do
	for step in  2.0  1.0 0.5  ; do
	    for two_factor in 10 ; do
	    	echo $ndim $step $two_factor $nsub
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 5
		time mpirun -n $wsize python -u  funnelmpi_multistep.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 10000  --nsub $nsub --lpath 10
	    done 
	done
    done
done

