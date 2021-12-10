#!/bin/bash

#### runs malpem-proot in bulk

runs=./input_MRIs/sub-*/anat/sub-*.nii.gz

for fl in $runs
do
malpem-proot -i $fl -o ./finished_runs -t 10
wait
echo "${fl} completed."
done
