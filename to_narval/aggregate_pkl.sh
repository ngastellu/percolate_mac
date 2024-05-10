#!/bin/bash



ii=()

while IFS= read -r line; do
	ii+=( "$line" )
done < 'good_runs.txt'

for i in ${ii[@]}; do
	echo $i
	if [ ! -d "to_local/sample-${i}" ]; then
		mkdir -p "to_local/sample-${i}"
	fi
	cp sample-${i}/*pkl "to_local/sample-${i}"
done
