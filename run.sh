#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4

scriptpath=$(dirname $0)
if [ ${scriptpath:0:1} == '.' ]; then
        scriptpath=$PWD/$scriptpath
fi;

if [ "$1" == "" ]; then
    echo "Usage: $0 t1_mri_filename"
    exit 1;
fi

a=$1
ba=$(basename $a)
a=$(basename $a .gz)
a=$(basename $a .nii)
pth=$(dirname $1)

which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

cd $pth

if [ ! -f ${a}_mni0Rigid.txt ]; then
  echo "Missing $pth/${a}_mni0Rigid.txt"
  python ${scriptpath}/model_apply_head_and_onlyhead.py $ba
fi

antsApplyTransforms -i $a.nii.gz -r $scriptpath/symboxR2.nii.gz -t "${a}_mni0Rigid.txt" -o aff_${a}.nii.gz -u float
antsApplyTransforms -i $a.nii.gz -r $scriptpath/symboxR2.nii.gz -t $scriptpath/flip.itk.mat -t "${a}_mni0Rigid.txt" -o aff_sym_${a}.nii.gz -u float

python $scriptpath/apply_run.py ${a}.nii.gz

antsApplyTransforms -i aff_${a}_slabroi.nii.gz -r $1 -o ${a}_slab_roiL.nii.gz -t [ "${a}_mni0Rigid.txt",1]  -u uchar -n MultiLabel[0.1]
antsApplyTransforms -i aff_sym_${a}_slabroi.nii.gz -r $1 -o ${a}_slab_roiR.nii.gz -t [ "${a}_mni0Rigid.txt",1] -t $scriptpath/flip.itk.mat -u uchar -n MultiLabel[0.1]

/bin/rm  aff_${a}.nii.gz aff_sym_${a}.nii.gz 
/bin/rm aff_${a}_slabroi.nii.gz aff_sym_${a}_slabroi.nii.gz

echo ${a1} webp animated images for verification : aff_${a}*webp
echo ${a1} output files ":" ${a}_slab_roiL.nii.gz ${a}_slab_roiR.nii.gz