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
  python ${scriptpath}/model_apply_head_and_onlyhead.py $ba
fi

python $scriptpath/apply_run.py ${a}.nii.gz

#/bin/rm ${a}_roiLeft.nii.gz ${a}_roiRightSym.nii.gz
cat << END > ${a}_index_webp.html
Image: ${a}<br/>
Left side <br/>
<img src=${a%%.nii.gz}_roiLeft_COR.webp>
<img src=${a%%.nii.gz}_roiLeft_AX.webp><br/>
Right side (mirrored)<br/>
<img src=${a%%.nii.gz}_roiRightSym_AX.webp>
<img src=${a%%.nii.gz}_roiRightSym_COR.webp><br/>
<br/>${a%%.nii.gz}_masseter_volumesLR.csv: (mm3)
<pre>
END
cat ${a%%.nii.gz}_masseter_volumesLR.csv >> ${a}_index_webp.html
echo "</pre>">> ${a}_index_webp.html
