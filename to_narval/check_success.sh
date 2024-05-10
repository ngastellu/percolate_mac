#!/bin/bash

outfile='good_runs.txt'

lbls=($(seq 0 131))

if [ -f $outfile ]; then
	rm $outfile
fi

for i in ${lbls[@]}; do
	x=$(ls -l sample-${i}/*pkl | wc -l)
	if [ $x -eq 40 ]; then
		echo $i >> $outfile
	fi
done
