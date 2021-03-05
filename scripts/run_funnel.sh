#!/bin/bash

for ndim in 2 5 10 20; do
    for step in 0.5 0.1 0.05 0.01 0.005; do
	echo $ndim $step
	python funnelpool.py --ndim $ndim --step_size $step --nsamples 10000 --nchains 20 --nparallel 20
    done
done
