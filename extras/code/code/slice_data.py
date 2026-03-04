import matplotlib.pyplot as plt
import nibabel as niib
import numpy as np
import SimpleITK as sitk


img_path = "/gpfs/work1/0/prjs0971/glioseg/data/scan20_1p/Patients/EGD-0004/NIFTI/T1.nii.gz"
img = sitk.ReadImage(img_path)

img_arr = sitk.GetArrayFromImage(img)
img_arr_nib = niib.load(img_path).get_fdata()
# img_arr_nib = np.transpose(img_arr_nib, (2, 1, 0))
print(img_arr.shape)
print(img_arr_nib.shape)
plt.figure()
plt.imshow(np.flipud(img_arr[11,:,:]), cmap='gray')
plt.savefig("slice.png")
plt.close()
plt.figure()
plt.imshow(np.rot90(img_arr_nib[:,:,11]), cmap='gray')
plt.savefig("slice_nib.png")
plt.close()
