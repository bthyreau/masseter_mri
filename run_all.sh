#!/bin/bash

scriptpath=$(dirname $0)
if [ "$1" == "" ]; then
	echo "Usage: run_all.sh file1 file2 ..."
	echo "       where each file is a T1-weighted head MRI in nifti format "
	exit 1
fi

outfiles=()
for a in $@; do
	if [ ${a} != ${a/slab_roi/} ]; then echo "ignoring output file ${a}"; continue; fi
	if [ ${a} != ${a/brain_mask/} ]; then echo "ignoring output file ${a}"; continue; fi
	$scriptpath/run.sh ${a}
	outfiles+=(${a})
done

pth=$(dirname ${a})
for a in ${outfiles[@]}; do cat ${a/.nii.gz/_index_webp.html}; echo "<hr>"; done > "${pth}/index.html"
