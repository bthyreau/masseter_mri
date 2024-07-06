#!/bin/bash

scriptpath=$(dirname $0)
if [ "$1" == "" ]; then
	echo "Usage: run_all.sh file1 file2 ..."
	echo "       where each file is a T1-weighted head MRI in nifti format "
	exit 1
fi

outfiles=()
for a in $@; do
	if [ ${a} != ${a/slab_roi/} ]; then echo "ignoring file ${a}"; continue; fi
	if [ ${a} != ${a/brain_mask/} ]; then echo "ignoring file ${a}"; continue; fi
	$scriptpath/run.sh ${a}
	outfiles+=(${a})
done

pth=$(dirname ${a})
for a in ${outfiles[@]}; do cat ${a/.nii.gz/_index_webp.html}; echo "<hr>"; done > "${pth}/index.html"
echo "filename,left_masseter_volume,right_masseter_volume,left_masseter_inslab_volume,right_masseter_inslab_volume" > "${pth}/all_masseter_volumes.csv"
for a in ${outfiles[@]}; do echo -n "$a,"; tail -n 1 "${a/.nii.gz/_masseter_volumesLR.csv}"; done >> "${pth}/all_masseter_volumes.csv"