from pathlib import Path

import numpy as np
import SimpleITK as sitk


def relabel_incorrect_necrosis(
    mask: sitk.Image,
    flair_scan: sitk.Image,
    csf_segmentation: sitk.Image | None,
    volume_threshold: int = 50,
    volume_threshold_pct: float = 15.0,
) -> sitk.Image:
    """
    Relabel necrosis in tumor segmentation, either with or without cysts.
    Args:
        mask (sitk.Image): Input segmentation mask (labels: 1=edema, 2=NCR/NET, 3=ET).
        flair_scan (sitk.Image): FLAIR scan.
        csf_segmentation (sitk.Image or None): CSF mask.
        volume_threshold (int): Volume threshold for early exists / cleanup.

    Returns:
        sitk.Image: Relabeled tumor segmentation mask (labels: 1=NET/edema, 2=NCR, 3=ET, 4=CC).
    """
    mask_array = sitk.GetArrayFromImage(mask)
    new_mask = mask_array.copy()

    ## Check volumes vs threshold
    if np.sum(mask_array == 3) <= volume_threshold:
        # if small ET: all -> label 1
        mask_array[mask_array > 0] = 1
        out = sitk.GetImageFromArray(mask_array)
        out.CopyInformation(mask)
        return out

    if np.sum(mask_array == 2) <= volume_threshold:
        # if small NCR: NCR -> label 1
        mask_array[mask_array == 2] = 1
        out = sitk.GetImageFromArray(mask_array)
        out.CopyInformation(mask)
        return out

    ## Enclosed necrosis check
    enhancing_img = sitk.BinaryThreshold(mask, 3, 3, 1, 0)
    enhancing_array = sitk.GetArrayFromImage(enhancing_img).astype(bool)

    # Close ET & fill holes
    closed = sitk.BinaryMorphologicalClosing(
        enhancing_img, 3 * [10]
    )
    filled = sitk.BinaryFillhole(closed, fullyConnected=True)
    filled_arr = sitk.GetArrayFromImage(filled).astype(bool)

    # Find cavities: filled ET - original ET
    cavities = filled_arr & (~enhancing_array)
    new_mask[cavities] = 10  # temporarily relabeled, to "protect" from cyst code

    ## FLAIR-based (necrotic) cyst check
    flair_array = sitk.GetArrayFromImage(flair_scan)
    edema_vals = flair_array[mask_array == 1]

    necrosis_mask = new_mask == 2

    csf_array = sitk.GetArrayFromImage(csf_segmentation)
    csf_vals = flair_array[csf_array > 0]

    if len(csf_vals) == 0 or len(edema_vals) == 0:
        print("Warning: empty CSF or edema reference, skipping necrotic cyst relabeling")
        new_mask[necrosis_mask] = 1
    else:
        # On remaining "necrosis": keep as label 2 if CSF < FLAIR int < edema, else label 1
        edema_qlo = np.percentile(edema_vals, 5)
        csf_qhi = np.percentile(csf_vals, 95)
        cystic_necrosis = (flair_array >= csf_qhi) & (flair_array <= edema_qlo)
        new_mask[necrosis_mask & (~cystic_necrosis)] = 1

        # On remaining "necrosis": check if necrotic cyst if > 15% of tumor volume
        tumor_volume = np.sum(mask_array > 0)

        necrosis_img = sitk.GetImageFromArray((new_mask == 2).astype(np.uint8))
        necrosis_img.CopyInformation(mask)
        cc = sitk.ConnectedComponent(necrosis_img)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)
        cc_arr = sitk.GetArrayFromImage(cc)

        for lbl in stats.GetLabels():
            if lbl == 0:
                continue
            blob_vol = stats.GetNumberOfPixels(lbl)
            if (
                100.0 * blob_vol / tumor_volume
            ) < volume_threshold_pct:
                blob_mask = cc_arr == lbl
                new_mask[blob_mask & (new_mask == 2)] = 1

        # Hole filling of cystic necrosis components
        necrosis_img_2 = sitk.GetImageFromArray((new_mask == 2).astype(np.uint8))
        necrosis_img_2.CopyInformation(mask)
        filled_necrosis = sitk.BinaryFillhole(necrosis_img_2, fullyConnected=True)
        new_mask[sitk.GetArrayFromImage(filled_necrosis).astype(bool)] = 2

    ## Combine all components
    new_mask[enhancing_array] = 3  # original ET

    # Relabel label 10 to necrosis per connected component, if at least 50% of border is ET
    label10_mask = new_mask == 10

    if np.any(label10_mask):
        cc10 = sitk.ConnectedComponent(sitk.GetImageFromArray(label10_mask.astype(np.uint8)))
        cc10_arr = sitk.GetArrayFromImage(cc10)

        for component_id in np.unique(cc10_arr)[1:]:
            component_mask = cc10_arr == component_id
            component_img = sitk.GetImageFromArray(component_mask.astype(np.uint8))
            dil = sitk.BinaryDilate(component_img, [1, 1, 1])
            dil_arr = sitk.GetArrayFromImage(dil).astype(bool)

            boundary = dil_arr & (~component_mask)

            if np.sum(boundary) == 0:
                perc = 0.0
            else:
                perc = 100.0 * np.sum(new_mask[boundary] == 3) / np.sum(boundary)

            # if <50% of boundary is ET -> ED/NET
            if perc >= 50:
                new_mask[component_mask] = 2
            else:
                new_mask[component_mask] = 1

    # Volume check on necrosis / cysts (to remove possible noise)
    if np.sum(new_mask == 2) <= volume_threshold:
        # if small NCR: NCR -> label 1
        new_mask[new_mask == 2] = 1
    if np.sum(new_mask == 4) <= volume_threshold:
        # if small CC: CC -> label 1
        new_mask[new_mask == 4] = 1

    out = sitk.GetImageFromArray(new_mask.astype(np.uint8))
    out.CopyInformation(mask)
    return out

if __name__ == "__main__":

    original_data_dir = Path("/scratch/radv/share/glioseg/skull_stripped_scans_2/patients")

for patient in original_data_dir.iterdir(): 
    
    best_gs_mask_file = patient / f"{patient.name}_brainles" / "raw_bet_mni152" / "MASK_best_GS.nii.gz"
    brain_mask_dir = patient / f"{patient.name}_brainles" / "brain_extracted_mni152" / "atlas__t1c_brain_mask.nii.gz"
    flair_im_dir = patient / f"{patient.name}_brainles" / "raw_bet_mni152" / f"{patient.name}_fla_bet.nii.gz"

    best_gs_mask = sitk.ReadImage(str(best_gs_mask_file), sitk.sitkUInt8)
    brain_mask = sitk.ReadImage(str(brain_mask_dir), sitk.sitkUInt8)
    flair_im = sitk.ReadImage(str(flair_im_dir))
    tumor_mask = sitk.Cast(sitk.ReadImage(str(best_gs_mask_file), sitk.sitkUInt8) > 0, sitk.sitkUInt8)

    brain_minus_tumor = sitk.And(brain_mask, sitk.InvertIntensity(tumor_mask, maximum=1))

    flair_masked = sitk.Mask(flair_im, brain_minus_tumor)

    vals = sitk.GetArrayFromImage(flair_masked)
    vals = vals[vals > 0]

    if len(vals) == 0:
        print(
            f"Warning: empty brain minus tumor mask for patient {patient.name}, creating empty CSF segmentation."
        )
        csf_segmentation = sitk.Image(flair_im.GetSize(), sitk.sitkUInt8)
        csf_segmentation.CopyInformation(flair_im)
    else: 

        p5 = float(np.percentile(vals, 5))
        min_val = float(np.min(vals))

        csf_raw = sitk.BinaryThreshold(flair_masked, min_val, p5, 1, 0)
        csf_segmentation = sitk.Mask(csf_raw, brain_minus_tumor)
        csf_segmentation.CopyInformation(flair_im)

    relabeled_mask = relabel_incorrect_necrosis(
        mask = best_gs_mask, 
        flair_scan = flair_im, 
        csf_segmentation = csf_segmentation
    )

    sitk.WriteImage(relabeled_mask, str(patient / f"{patient.name}_brainles" / "raw_bet_mni152" / "MASK_best_GS_relabelled.nii.gz"))