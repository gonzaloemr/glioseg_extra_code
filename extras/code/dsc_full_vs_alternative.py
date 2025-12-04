import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from nibabel.orientations import aff2axcodes


# def dice_score(gt,seg,labels):
#     seg = np.round(seg)
#     gt = np.round(gt)

#     dice = []
#     for label in labels:
#         num = float(np.sum(seg[gt == label] == label) * 2.0)
#         den = float(np.sum(seg == label) + np.sum(gt == label))
#         if den == 0:
#             d = 1
#         else:
#             d = round(num / den,4)
#         dice.append(d)

#     return dice

# def correct_orientation(image_arr):
#     transform = nib.orientations.axcodes2ornt(("R", "A", "S"))
#     print(transform)
#     img = nib.orientations.apply_orientation(image_arr, transform)
#     img_final = img
#     return img_final

def dice_score(mask,pred, classes):

    dice_scores = []

    for class_id in classes:
      mask_class = (mask == class_id).astype(int)
      pred_class = (pred == class_id).astype(int)
      if mask_class.sum() != 0 and mask_class.sum() <5:
         print(mask_class)
         dice = np.nan
      elif mask_class.sum() == 0:
         if pred_class.sum() <=5:
            dice = 1
         else:
            dice = 0
      else:
        intersection = 2.0 * (pred_class * mask_class).sum()
        union = pred_class.sum() + mask_class.sum()

        if union == 0: #Ground truth has 0 cases, prediction has 0 cases -> agreement, 1! 
            dice = 1
        else:       
            dice = round(intersection / union, 4)
      dice_scores.append(dice)

    return dice_scores

data_path = "/gpfs/work1/0/prjs0971/glioseg/data/scan20_1p/Patients/EGD-0004/SEGMENTATIONS/ATLAS_SRI24"

full_mode = nib.load(os.path.join(data_path,"full_mask_tumor_scan2020.nii.gz")).get_fdata()
full_mode_na = nib.load(os.path.join(data_path,"full_na_tumor_scan2020.nii.gz")).get_fdata()
lite_mode = nib.load(os.path.join(data_path,"lite_tumor_scan2020.nii.gz")).get_fdata()
lite_mode_na = nib.load(os.path.join(data_path,"lite_na_tumor_scan2020.nii.gz")).get_fdata()

print(dice_score(full_mode,full_mode,[1,2,4]))
print(dice_score(full_mode,full_mode_na,[1,2,4]))
print(dice_score(full_mode,lite_mode,[1,2,4]))
print(dice_score(full_mode,lite_mode_na,[1,2,4]))


# data_path_1p = "/projects/0/prjs0971/glioseg/data/scan20_1p/Patients"
# data_path_2p = "/projects/0/prjs0971/glioseg/data/scan20_2p/Patients"
# data_path_4p = "/projects/0/prjs0971/glioseg/data/scan20_4p/Patients"

# for patient in os.listdir(data_path_1p): 
#    scan_path_gt = nib.load(os.path.join(data_path_1p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020.nii.gz")).get_fdata()
#    scan_one_copy = nib.load(os.path.join(data_path_1p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_one_copy.nii.gz")).get_fdata()
#    print(dice_score(scan_one_copy, scan_path_gt, [1,2,4]))

# for patient in os.listdir(data_path_2p): 
#    scan_path_gt = nib.load(os.path.join(data_path_2p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020.nii.gz")).get_fdata()
#    scan_one_copy = nib.load(os.path.join(data_path_2p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_one_copy.nii.gz")).get_fdata()
#    print(dice_score(scan_one_copy, scan_path_gt, [1,2,4]))

# for patient in os.listdir(data_path_4p): 
#    print(patient)
#    scan_path_gt = nib.load(os.path.join(data_path_4p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020.nii.gz")).get_fdata()
#    scan_one_copy = nib.load(os.path.join(data_path_4p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_one_copy.nii.gz")).get_fdata()
#    print(dice_score(scan_one_copy, scan_path_gt, [1,2,4]))

# data_ex = nib.load("/projects/0/prjs0971/glioseg/data/14_patients/Patients/EGD-1038/SEGMENTATIONS/ATLAS_SRI24/mask_tumor_scan2020.nii.gz")
# # data_converted = correct_orientation(data_ex.get_fdata())
# # print(data_converted.shape)
# # data_ex_np = np.transpose(data_ex.get_fdata(),(1,0,2))
# data_exp_np = data_ex.get_fdata()
# # data_ex_np = data_ex.get_fdata()[:,:,47]
# # data_ex_np = data_ex_np.T   
# print(np.unique(data_exp_np, return_counts=True))
# print(np.where(data_exp_np==1.0))
# print(np.where(data_converted==1.0))
# print(aff2axcodes(data_ex.affine))
# plt.imshow(data_ex_np[:,:,48],cmap="gray")
# plt.imshow(data_converted[:,:,48],cmap="gray")
# plt.savefig("EGD-1038.png")



# data_path_14p = "/projects/0/prjs0971/glioseg/data/14_patients/Patients"
# dice_self =[]
# dice_full_no_aug = []
# dice_lite_aug = []
# dice_lite_no_aug = []
# subjects = []

# for patient in os.listdir(data_path_14p):
   
#     if os.path.isdir(os.path.join(data_path_14p,patient)):
#         print(patient)
#         scan_path_gt = nib.load(os.path.join(data_path_14p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020.nii.gz")).get_fdata()  
#         scan_path_full_no_aug = nib.load(os.path.join(data_path_14p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_full_no_aug.nii.gz")).get_fdata()
#         scan_path_lite_aug = nib.load(os.path.join(data_path_14p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_lite_aug.nii.gz")).get_fdata()
#         scan_path_lite_no_aug = nib.load(os.path.join(data_path_14p,patient,"SEGMENTATIONS","ATLAS_SRI24","mask_tumor_scan2020_lite_no_aug.nii.gz")).get_fdata()

#         print("Ground truth", np.unique(scan_path_gt,return_counts=True))
#         print("Full no aug", np.unique(scan_path_full_no_aug,return_counts=True))
#         print("Lite aug", np.unique(scan_path_lite_aug,return_counts=True))
#         print("Lite no aug", np.unique(scan_path_lite_no_aug,return_counts=True))
#         print('-'*100)

#         classes = [1,2,4]
#         subjects.append(patient)

#         dice_full_no_aug.append(dice_score(scan_path_gt, scan_path_full_no_aug, classes))
#         dice_lite_aug.append(dice_score(scan_path_gt, scan_path_lite_aug, classes))
#         dice_lite_no_aug.append(dice_score(scan_path_gt, scan_path_lite_no_aug, classes))
#         dice_self.append(dice_score(scan_path_gt,scan_path_gt,classes))

    
# subjects += ["Mean", "Median", "Standard deviation"]

# dice_full_no_aug_df = pd.DataFrame(dice_full_no_aug,columns=["Full no aug - necrotic core", "Full no aug - edema", "Full no aug - Enhancing tumor"])
# mean_dice_full_no_aug = round(dice_full_no_aug_df.mean(axis=0),3)
# median_dice_full_no_aug = round(dice_full_no_aug_df.median(axis=0),3)
# std_dice_full_no_aug = round(dice_full_no_aug_df.std(axis=0),3)
# dice_full_no_aug_df = pd.concat([dice_full_no_aug_df, mean_dice_full_no_aug.to_frame().T, median_dice_full_no_aug.to_frame().T,std_dice_full_no_aug.to_frame().T])
# dice_full_no_aug_df.index = subjects
# # dice_full_no_aug_df.to_csv(os.path.join(data_path_14p,"dice_full_no_aug.csv"))

# dice_lite_aug_df = pd.DataFrame(dice_lite_aug,columns=["Lite aug - necrotic core", "Lite aug - edema", "Lite aug - Enhancing tumor"])
# mean_dice_lite_aug = round(dice_lite_aug_df.mean(axis=0),3)
# median_dice_lite_aug = round(dice_lite_aug_df.median(axis=0),3)
# std_dice_lite_aug = round(dice_lite_aug_df.std(axis=0),3)
# dice_lite_aug_df = pd.concat([dice_lite_aug_df, mean_dice_lite_aug.to_frame().T, median_dice_lite_aug.to_frame().T,std_dice_lite_aug.to_frame().T])
# dice_lite_aug_df.index = subjects
# # dice_lite_aug_df.to_csv(os.path.join(data_path_14p,"dice_lite_aug.csv"))

# dice_lite_no_aug_df = pd.DataFrame(dice_lite_no_aug,columns=["Lite no aug - necrotic core", "Lite no aug - edema", "Lite no aug - Enhancing tumor"])
# mean_dice_lite_no_aug = round(dice_lite_no_aug_df.mean(axis=0),3)
# median_dice_lite_no_aug = round(dice_lite_no_aug_df.median(axis=0),3)
# std_dice_lite_no_aug = round(dice_lite_no_aug_df.std(axis=0),3)
# dice_lite_no_aug_df = pd.concat([dice_lite_no_aug_df, mean_dice_lite_no_aug.to_frame().T, median_dice_lite_no_aug.to_frame().T,std_dice_lite_no_aug.to_frame().T])
# dice_lite_no_aug_df.index = subjects
# # dice_lite_no_aug_df.to_csv(os.path.join(data_path_14p,"dice_lite_no_aug.csv"))


# # dice_self_df = pd.DataFrame(dice_self,columns=["Self - necrotic core", "Self - edema", "Self - Enhancing tumor"])
# # mean_dice_self = round(dice_self_df.mean(axis=0),3)
# # median_dice_self = round(dice_self_df.median(axis=0),3)
# # std_dice_self = round(dice_self_df.std(axis=0),3)
# # dice_self_df = pd.concat([dice_self_df, mean_dice_self.to_frame().T, median_dice_self.to_frame().T,std_dice_self.to_frame().T])
# # dice_self_df.index = subjects

# dice_scan20 = pd.concat([dice_full_no_aug_df, dice_lite_aug_df, dice_lite_no_aug_df], axis=1)
# dice_scan20.to_csv(os.path.join(data_path_14p,"dice_scan20.csv"))

# # mean_dice_full_no_aug = np.mean(dice_full_no_aug,axis=0)
# # dice_full_no_aug.append(mean_dice_full_no_aug)
# # mean_dice_lite_aug = np.mean(dice_lite_aug,axis=0)
# # dice_lite_aug.append(mean_dice_lite_aug)
# # mean_dice_lite_no_aug = np.mean(dice_lite_no_aug,axis=0)
# # dice_lite_no_aug.append(mean_dice_lite_no_aug)

# # median_dice_full_no_aug = np.median(dice_full_no_aug,axis=0)
# # dice_full_no_aug.append(median_dice_full_no_aug)
# # median_dice_lite_aug = np.median(dice_lite_aug,axis=0)
# # dice_lite_aug.append(median_dice_lite_aug)
# # median_dice_lite_no_aug = np.median(dice_lite_no_aug,axis=0)
# # dice_lite_no_aug.append(median_dice_lite_no_aug)

# # std_dice_full_no_aug = np.std(dice_full_no_aug,axis=0)
# # dice_full_no_aug.append(std_dice_full_no_aug)
# # std_dice_lite_aug = np.std(dice_lite_aug,axis=0)
# # dice_lite_aug.append(std_dice_lite_aug)
# # std_dice_lite_no_aug = np.std(dice_lite_no_aug,axis=0)
# # dice_lite_no_aug.append(std_dice_lite_no_aug)

# # subjects += ["Mean", "Median", "Std"]

# # results = pd.DataFrame.from_dict({"Subjects": subjects,
# #                                    "Full mode with no augmentation": dice_full_no_aug, "Lite mode with augmentation": dice_lite_aug, "Lite mode with no augmentation": dice_lite_no_aug})
# # results.to_csv(os.path.join(data_path_14p,"results_dice_comparison.csv"))