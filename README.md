## Segments the masseter muscle on 3D head MRI

![anim](https://github.com/bthyreau/masseter_mri/assets/590921/5ec13982-b3d0-4145-91a4-bc39abd179b5)

*Masticatory muscle (brown) and cross-section plane (green)*


This software expects a (mostly isotropic) 3D image of the head covering the masticatory muscle. It is based on a machine-learning (ConvNet) model trained on 3D T1-weighted MRIs of a cohort of subjects aged 50+, although it should work well on any population. 

*More detail about the model and training approach (will be) described in Chen et al.*

### Requirements
The input MRI should be a T1-weighted image of the head, with a field-of-view that extends low enough to cover the masseter muscle area, and in nifti format. If your data is DICOM, you can easily convert it to nifti using [dcm2niix](https://github.com/rordenlab/dcm2niix). 

The program uses [PyTorch](https://pytorch.org/). A CPU version is sufficient, no GPU necessary

### Installation
To install this program using the uv packaging tool ( https://docs.astral.sh/uv/ ), you can simply run `uv sync` in this repository, and it should take care of downloading and installing the dependencies.

Alternatively, you'll need to configure a python 3 environment on your machine. For e.g. to install using pip3 (works both Linux and MacOS)
```
pip3 install scipy nibabel pillow
pip3 install torch
```
Note that on Linux this may download the unnecessary CUDA version of PyTorch, refers to https://pytorch.org/get-started/locally/ for more precise control.


### Single image processing
You can run the segmentation of a single image simply by calling

```./masseter_run.sh example_t1wi.nii.gz```

The script masseter_run.sh, or a symlink to it, can safely be made available in the $PATH for convenience.

### Cohort processing

Multiple images can be analysed with the script:
```./masseter_run_all.sh *.nii.gz```

In this case, the script will creates (*in the current working directory*) a ```index.html``` page for easy visualization of all the **segmentation masks**, and a summary csv table with the **volume** of the muscle (left and right) and the interesection with the plane. 

For each file, the output files example_t1wi_slab_roiL.nii.gz are example_t1wi_slab_roiR.nii.gz, each containing the muscle on its respective side, as well as a cross-section plane mask mapping 2.5cm to 3cm below the estimated Camper plane.  For quick visualization purpose and quality checks, some animation similar to the one above are generated in .webp format.
