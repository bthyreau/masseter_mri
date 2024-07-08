## Segments the masseter muscle on T1-weighted head MRI

![anim](https://github.com/bthyreau/masseter_mri/assets/590921/5ec13982-b3d0-4145-91a4-bc39abd179b5)

*Masticatory muscle (brown) and cross-section plane (green)*


This software expects a T1-weighted 3D image of the head covering the masticatory muscle. It is based on a machine-learning (ConvNet) model trained on a cohort of 50+ aged subject, although it should work well on any population. 

*More detail about the model and training approach (will be) described in Liu et al.*

### Installation and requirements
The input should be in nifti format. If you have DICOM data, you can convert it to nifti using [dcm2niix](https://github.com/rordenlab/dcm2niix)

The program needs to run on python3 (with packages `scipy`, `PIL` and `nibabel`, all available on _pip_ or _conda_) and a [pytorch](https://pytorch.org/get-started/locally/) environment.

### Single image processing
You can run the segmentation on a single image simply by calling

```./run.sh head_mri.nii.gz```

The output files head_mri_slab_roiL.nii.gz are head_mri_slab_roiR.nii.gz, each containing the muscle on its respective side, as well as a cross-section plane mask mapping 2.5cm to 3cm below the estimated Camper plane.  For quick visualization purpose and quality checks, some animation similar to the one above are generated in .webp format.

### Cohort processing

Multiple images can be analysed with:
```./run_all.sh *.nii.gz```

In this case, the script outputs an ```index.html``` page for easy visualization of all the **segmentation masks**, and a summary csv table with the **volume** of the muscle (left and right) and the interesection with the plane.

