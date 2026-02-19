import numpy as np


softmax_file_dir_example = "/scratch/radv/share/glioseg/GT_Vera/Patients/IM0015/SEGMENTATIONS/ATLAS/SOFTMAX/mask_tumor_hdglio.npz"
data = np.load(softmax_file_dir_example)
data_softmax = data['softmax']
print(data_softmax.shape)