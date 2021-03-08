#!/bin/bash

for ndim in 10 20; do
#for ndim in 2 5 10 20; do
    for step in  0.2 0.1 0.05 0.01 0.005; do
	echo $ndim $step
	#python funnelstan.py --ndim $ndim --step_size $step --nsamples 5000 --nchains 20 --nparallel 20
    done
done


for ndim in 2 5 ; do
#for ndim in 2 5 10 20; do
    for step in  0.2 0.1 0.05 0.01 0.005; do
	echo $ndim $step
	python funnelpool.py --ndim $ndim --step_size $step --nsamples 5000 --nchains 20 --nparallel 20
    done
done


#for ndim in 2 5 10 20; do
for ndim in  2 5 ; do
    for step in  0.2 0.1 0.05 0.01; do
	for two_factor in 2 5 10; do
	    echo $ndim $step $two_factor
	    python funnelpool2step.py --ndim $ndim --step_size $step --two_factor $two_factor --nsamples 5000 --nchains 20 --nparallel 20
	done
    done
done

