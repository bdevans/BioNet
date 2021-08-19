#!/usr/bin/env bash

ECOSET_DIR=${1:-../ecoset}
declare -A categories=( ["airplane"]="0356_airplane" ["automobile"]="0009_car" ["bird"]="0085_bird" ["cat"]="0148_cat" ["deer"]="1028_deer" ["dog"]="0039_dog" ["frog"]="0164_frog" ["horse"]="0050_horse" ["ship"]="0077_ship" ["truck"]="0149_truck" )

for dir in train val test;
do
	for category in "${!categories[@]}";
	do
		ln -s "$ECOSET_DIR"/"$dir"/"${categories[$category]}" "$dir"/"$category"
	done
done
