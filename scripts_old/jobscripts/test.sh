#!/bin/bash

for i in "1 a" "2 b" "3 c"; do a=( $i ); echo "${a[1]} ${a[0]}"; echo "${a[0]}"; done



ndim=100
two_factor=5
lpath=30

for i in "4 2.0" "3 1.0" "4 1.0"; do
    x=$i
    nsub=${x[0]}
    step=${x[1]}
    echo $x
    echo $nsub $step
done
