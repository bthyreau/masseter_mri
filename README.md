## Segments the masseter muscle on T1-weighted head MRI

![anim](https://github.com/bthyreau/masseter_mri/assets/590921/5ec13982-b3d0-4145-91a4-bc39abd179b5)

Masticatory muscle (brown) and cross-section plane (green) are highlighted

### Install
This software requires a T1-weighted 3D image of the head, covering the masticatory muscle.

The input should be in nifti format. If you have DICOM data, you can convert it to nifti using [dcm2niix](https://github.com/rordenlab/dcm2niix)

The program needs to run on a python3 and [pytorch](https://pytorch.org/get-started/locally/) environment. Then you can run the segmentation simply by calling

```./run.sh head_mri.nii.gz```


## output
The output files head_mri_slab_roiL.nii.gz are head_mri_slab_roiR.nii.gz, each containing the muscle on its respective side, as well as the cross-section plane.

Volume are summarized in head_mri_slabroi_volumesLR.txt (resp, Muscle-L, slab-L, Muscle-R, slab-R)

For quick visualization purpose and quality checks, some animation similar to the one above are generated in .webp format.


