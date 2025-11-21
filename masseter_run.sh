#!/bin/bash
r0=$(realpath $0)
PATH=$(dirname $r0)/.venv/bin/:$(dirname $r0)/venv/bin/:$PATH

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

cd $pth

if [ ! -f ${a}_mni0Rigid.txt ]; then
  python3 $(dirname $r0)/model_apply_head_and_onlyhead.py $ba
fi

python3 $(dirname $r0)/apply_run.py ${ba}

#/bin/rm ${a}_roiLeft.nii.gz ${a}_roiRightSym.nii.gz
cat << END > ${a}_index_webp.html
<pre> <table style="color:white;background-color:black"> <thead> <tr><th colspan="4">${pth}/${a}</th></tr><tr> <th colspan="2">Left Side</th> <th colspan="2">Right Side (mirrored)</th> </tr> </thead>
<tbody> <tr>
  <td><img src="${a}_roiLeft_COR.webp" alt="Left COR"></td>
  <td><img src="${a}_roiLeft_AX.webp" alt="Left AX"></td>
  <td><img src="${a}_roiRightSym_COR.webp" alt="Right Sym COR"></td>
  <td><img src="${a}_roiRightSym_AX.webp" alt="Right Sym AX"></td>
</tr> </tbody> </table>
volume (mm3)
END
cat ${a}_masseter_volumesLR.csv >> ${a}_index_webp.html
echo "</pre>">> ${a}_index_webp.html
