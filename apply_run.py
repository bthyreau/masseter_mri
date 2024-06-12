import torch
from torch import nn
import torch.nn.functional as F
import nibabel, os
import numpy as np
import scipy.ndimage



import glob
from PIL import Image
     
#allf = sorted(glob.glob("aff_*.nii.gz"))
#allf = [x for x in allf if not x.endswith("_slabroi.nii.gz")]
def make_webps(fn):
    print("Generating webp for", fn)
    m = np.asarray(nibabel.load(fn.replace(".nii.gz", "_slabroi.nii.gz")).dataobj)
    d = nibabel.load(fn).get_fdata(dtype=np.float32)
    
    d -= d.min()
    d /= (d.max() / 300.).clip(0, 255)
    #print(d.min(), d.max())
    
    min_slice, max_slice = np.where(np.diff(((m == 1) ).sum(2).sum(0) > 0 ))[0][:2]
    out = []
    for num in range(min_slice, max_slice, 2):
        i = Image.fromarray(( d[20:130, num, :]).clip(0, 200) / 200 * 255).convert("RGB")
        ma = m[20:130, num, :]
        tm = ((ma == 1) | (ma == 2)).astype(np.uint8)
        im = Image.fromarray(np.dstack([(ma==1)*255, tm*128, np.ones_like(ma)*16]).astype(np.uint8), mode="RGB")
        i.paste(im, mask=Image.fromarray(tm*56))
        out.append(i)

    out[0].save('%s_AX.webp' % fn, save_all=True, append_images=out[1:], loop=0, duration=80)

    
    min_slice, max_slice = np.where(np.diff(((m == 1) ).sum(1).sum(0) > 0 ))[0][:2]
    out = []
    for num in range(min_slice - 1, max_slice + 1, 1):
        i = Image.fromarray(np.rot90(d[20:130, :, num]).clip(0, 200) / 200 * 255 ).convert("RGB")
        ma = np.rot90(m[20:130, :, num])
        tm = ((ma == 1) | (ma == 2)).astype(np.uint8)
        im = Image.fromarray(np.dstack([(ma==1)*255, tm*128, np.zeros_like(ma)]).astype(np.uint8), mode="RGB")
        i.paste(im, mask=Image.fromarray(tm*56))
        out.append(i)

    out[0].save('%s_COR.webp' % fn, save_all=True, append_images=out[1:], loop=0, duration=80)
    print(" Done")


class JawsModel2(nn.Module):
    def __init__(self):
        super(JawsModel2, self).__init__()
        self.conv0a = nn.Conv3d(1, 12, 3, padding=1)
        self.conv0b = nn.Conv3d(12, 12, 3, padding=1)
        self.bn0a = nn.BatchNorm3d(12)

        self.ma1 = nn.MaxPool3d(2)
        self.conv1a = nn.Conv3d(12, 12, 3, padding=1)
        self.conv1b = nn.Conv3d(12, 12, 3, padding=1)
        self.bn1a = nn.BatchNorm3d(12)

        self.ma2 = nn.MaxPool3d(2)
        self.conv2a = nn.Conv3d(12, 16, 3, padding=1)
        self.conv2b = nn.Conv3d(16, 16, 3, padding=1)
        self.bn2a = nn.BatchNorm3d(16)

        self.ma3 = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3b = nn.Conv3d(32, 24, 3, padding=1)
        self.bn3a = nn.BatchNorm3d(24)

        # up

        self.conv2u = nn.Conv3d(24, 16, 3, padding=1)
        self.bn2u = nn.BatchNorm3d(16)
        self.conv2v = nn.Conv3d(16+0*16, 16, 3, padding=1)

        # up

        self.conv1u = nn.Conv3d(16, 12, 3, padding=1)
        self.bn1u = nn.BatchNorm3d(12)
        self.conv1v = nn.Conv3d(12+0*12, 12, 3, padding=1)

        # up

        self.conv0u = nn.Conv3d(12, 12, 3, padding=1)
        self.bn0u = nn.BatchNorm3d(12)
        self.conv0v = nn.Conv3d(12+0*12, 12, 3, padding=1)

        self.conv1x = nn.Conv3d(12, 3, 1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv0a(x))
        self.li0 = x = torch.relu(self.bn0a(self.conv0b(x)))

        x = self.ma1(x)
        x = torch.relu(self.conv1a(x))
        self.li1 = x = torch.relu(self.bn1a(self.conv1b(x)))

        x = self.ma2(x)
        x = torch.relu(self.conv2a(x))
        self.li2 = x = torch.relu(self.bn2a(self.conv2b(x)))

        x = self.ma3(x)
        x = torch.relu(self.conv3a(x))
        self.li3 = x = torch.relu(self.bn3a(self.conv3b(x)))

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = torch.relu(self.bn2u(self.conv2u(x)))

        x = x + self.li2
        x = torch.relu(self.conv2v(x))

        self.lo1 = x
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = torch.relu(self.bn1u(self.conv1u(x)))

        x = x + self.li1
        #with concatenation istead of add, it would be "x = torch.cat([x, self.li1], axis=1)", but the number of kernels must be changed in the definition
        x = torch.relu(self.conv1v(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        self.la1 = x

        x = torch.relu(self.bn0u(self.conv0u(x)))

        x = x + self.li0
        x = torch.relu(self.conv0v(x))

        self.out = x = self.conv1x(x)
        x = torch.sigmoid(x)
        return x

# Landmarks
net1 = JawsModel2()
scriptpath = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cpu")
net1.load_state_dict(torch.load(scriptpath + "/torchparams/myworking_model4.pt", map_location=device), strict=False)

device = "cpu"

import sys
fn = sys.argv[1]
assert fn.endswith(".nii.gz")


def read_mni0txt(filename_mni0txt):
    f3 = np.array([[1, 1, -1, -1],[1, 1, -1, -1], [-1, -1, 1, 1], [1, 1, 1, 1]]) # ANTs LPS
    p = np.array(open(filename_mni0txt).read().split("Parameters: ")[-1].split(), float)
    return np.r_[np.c_[ p[:9].reshape(3,3), p[9:] ], [[0,0,0,1]]] / f3

Mmni = read_mni0txt(fn.replace(".nii.gz", "_mni0Rigid.txt")) #"../../partialnii/hirosaki_H31120_mni0Affine.txt")





import sys
fn = sys.argv[1]
assert fn.endswith(".nii.gz")
img2 = nibabel.load(fn)

if nibabel.aff2axcodes(img2.affine) != ("P","S","R"):
    print("Reorienting internally")
    trn = nibabel.orientations.ornt_transform( nibabel.io_orientation(img2.affine), np.array([[1,-1], [2, 1], [0,1]]))
    img = nibabel.Nifti1Image(nibabel.orientations.apply_orientation(img2.dataobj, trn), img2.affine @ nibabel.orientations.inv_ornt_aff( trn, img2.shape)) #.to_filename("/tmp/test.nii.gz")
else:
    img = img2
#img = nibabel.load("/tmp/test.nii.gz")


# The T1-MRI needs to have a resolution of 96x96x64 for the landmarks model to work, resample along axis
i = np.mgrid[:img.shape[0]-1:96j, :img.shape[1]-1:96j, :img.shape[2]-1:64j ]
fake_affine = img.affine.copy()
assert nibabel.aff2axcodes(img.affine) == ("P","S","R"), "Voxel orientation error: the landmarks code function assume some orientation. The downsampling code should be fixed, but why bother"

fake_affine[:3,:3] *= i.reshape(3, -1).max(1) / [95, 95, 63]
d = scipy.ndimage.map_coordinates(img.get_fdata(dtype=np.float32), i, order=2)
del i
d -= d.mean()
d /= d.std()

with torch.no_grad():
    output = net1(torch.from_numpy(d.astype(np.float32)[None,None,:]).to(device))

output[output < .5] = 0
output = np.asarray(output)
centr_vox = np.array([scipy.ndimage.center_of_mass(m) + (1,) for m in output[0]])


if np.any(np.isnan(centr_vox)):
    print("Impossible to identify landmarks. Are you sure the masseter muscle is visible ?")



centr_mm = centr_vox @ fake_affine.T

print("The three landmarks coordinates:\n", centr_mm[:,:3])





####
#net2 = JawsModel()
net1.load_state_dict(torch.load(scriptpath + "/torchparams/_params_jaws3l2_00299_00000.pt", map_location=device))


#img = nibabel.load("../../partialnii/aff_sym_hirosaki_H31120.nii.gz")
#affsym_fn = "../../partialnii/aff_sym_hirosaki_H31120.nii.gz"
stats_vols = {}
for side in "L", "R":

    if side == "L":
        aff_fn = fn.replace(os.path.basename(fn), "aff_" + os.path.basename(fn))
    else:
        aff_fn = fn.replace(os.path.basename(fn), "aff_sym_" + os.path.basename(fn))

    print("Processing", aff_fn)
    img = nibabel.load(aff_fn)

    d = img.get_fdata(dtype=np.float32)
    d -= d.mean()
    d /= d.std()

    with torch.no_grad():
        output = net1(torch.from_numpy(d.astype(np.float32)[None,None,:]).to(device))

    output[output < .5] = 0
    output[output > .9] = 1
    output = np.asarray(output[0,0]) * 255
    segmap = output.astype(np.uint8)

    voxvol = np.abs(np.linalg.det(img.affine))

    roivol = segmap.sum() * voxvol / 255.
    print("Volume in seg ", img.get_filename(), roivol)
    stats_vols[(side, "r")] = roivol

    c = centr_mm[:,:3]
    v = np.cross(c[1] - c[0], c[2] - c[0])
    v /= np.linalg.norm(v)

    pts = np.indices(img.shape).reshape(3, -1).T
    pts = np.hstack((pts,np.ones((pts.shape[0],1))))
    #pts = pts @ img.affine.T
    maybeflipMni = np.diag([-1, 1,1,1]) if "_sym_" in img.get_filename() else np.diag([1,1,1,1])
    #pts = pts @ (Mmni @ img.affine).T
    pts = pts @ (Mmni @ maybeflipMni @ img.affine).T

    cpts = ( pts[:,:3]- c[0] )
    
    output = np.dot(cpts, v)
    output = output.reshape(img.shape)

    slab_starts_mm = +25
    slab_ends_mm = +30

    slab = (slab_starts_mm <= output ) & (output <= slab_ends_mm)

    roislabvol = (segmap * slab).sum() * voxvol / 255.
    print(" in slab", roislabvol)
    stats_vols[(side, "s")] = roislabvol

    if 0:
        nibabel.Nifti1Image(slab.astype(np.uint8), img.affine).to_filename(img.get_filename().replace(".nii.gz", "_slab.nii.gz"))
        nibabel.Nifti1Image(segmap.astype(np.uint8), img.affine).to_filename(img.get_filename().replace(".nii.gz", "_roi.nii.gz"))
    #print(img.get_filename().replace(".nii.gz", "_slab.nii.gz"))
    
    output_image = (segmap > 128).astype(np.uint8)
    output_image[slab == 1] += 2
    nibabel.Nifti1Image(output_image, img.affine).to_filename(img.get_filename().replace(".nii.gz", "_slabroi.nii.gz"))
    
    make_webps(aff_fn)
    #t fslview ../../partialnii/aff_sym_hirosaki_H31120.nii.gz /tmp/seg.nii /tmp/slab.nii &

open(fn.replace(".nii.gz", "_slabroi_volumesLR.txt"), "w").write("%d,%d,%d,%d\n" % (stats_vols["L", "r"], stats_vols["L", "s"], stats_vols["R", "r"], stats_vols["R", "s"]))



#sys.exit(1)


