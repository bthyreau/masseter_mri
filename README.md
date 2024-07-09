## Segments the masseter muscle on T1-weighted head MRI

![anim](https://github.com/bthyreau/masseter_mri/assets/590921/5ec13982-b3d0-4145-91a4-bc39abd179b5)

*Masticatory muscle (brown) and cross-section plane (green)*


This software expects a T1-weighted 3D image of the head covering the masticatory muscle. It is based on a machine-learning (ConvNet) model trained on a cohort of 50+ aged subject, although it should work well on any population. 

*More detail about the model and training approach (will be) described in Liu et al.*

### Installation and requirements
The input should be in nifti format. If you have DICOM data, you can convert it to nifti using [dcm2niix](https://github.com/rordenlab/dcm2niix)

The program needs to run on a [PyTorch](https://pytorch.org/get-started/locally/) python environment. A CPU version of Pytorch is sufficient.

For e.g. to install using pip3 (works both Linux and MacOS)
```
pip3 install scipy nibabel pillow
pip3 install torch
```
On Linux this may download the full CUDA by default, so refers to https://pytorch.org/get-started/locally/ on how to skip that, ie. `pip3 install torch --index-url https://download.pytorch.org/whl/cpu`)

Then, download or clone this repository, assuming it is in `./masseter_mri-main`, and runs the script at below:

### Single image processing
You can run the segmentation of a single image simply by calling

```/path/to/masseter_mri-main/run.sh example_t1wi.nii.gz```

where `/path/to/masseter_mri-main/` should be the full path to this masseter_mri-main repository

### Cohort processing

Multiple images can be analysed with the script:
```/path/to/masseter_mri-main/run_all.sh *.nii.gz```

In this case, the script will creates (in the current directory) a ```index.html``` page for easy visualization of all the **segmentation masks**, and a summary csv table with the **volume** of the muscle (left and right) and the interesection with the plane. 

For each file, the output files example_t1wi_slab_roiL.nii.gz are example_t1wi_slab_roiR.nii.gz, each containing the muscle on its respective side, as well as a cross-section plane mask mapping 2.5cm to 3cm below the estimated Camper plane.  For quick visualization purpose and quality checks, some animation similar to the one above are generated in .webp format.
