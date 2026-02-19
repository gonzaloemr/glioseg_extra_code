import itertools
import math

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from metrics_nnunet import compute_dice_score
from metrics_nnunet import extract_label
from metrics_nnunet import same_physical_space
from scipy.stats import pearsonr

import glioseg.constants as constants


def compute_mean_ensemble_segmentation(
    input_dir, gt_dir, segmentation_dict, labels, output_dir, relabelled: bool = False
):
    """
    Generate a mean ensemble segmentation for each patient by averaging model predictions voxelwise
    and discretizing according to label thresholds.

    Rules:
        - Mean < 0.5 → label 0
        - 0.5 ≤ Mean < 1.5 → label 1
        - 1.5 ≤ Mean < 2.5 → label 2
        - ≥ 2.5 → label 3

    Args:
        input_dir (Path): Parent directory with patient subfolders.
        segmentation_dict (dict): {method_name: suffix_string} same as in quality assessment.
        output_dir (Path): Directory where ensemble results will be saved.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_results = []

    for patient in input_dir.iterdir():
        segmentations_dir = patient.joinpath("SEGMENTATIONS", "NIFTI")
        seg_arrays = []
        reference_im = None

        # --- Load all model segmentations ---
        for method, seg_file_name in segmentation_dict.items():
            seg_file = segmentations_dir.joinpath(
                constants.TUMOR_MASK_NAME + seg_file_name + "_T1GD"
            ).with_suffix(constants.NIFTI_EXTENSION)

            # Handle relabelled versions
            if not seg_file.exists() and "_relabelled" in seg_file_name:
                seg_file_name = seg_file_name.replace("_relabelled", "")
                seg_file = segmentations_dir.joinpath(
                    constants.TUMOR_MASK_NAME + seg_file_name + "_T1GD"
                ).with_suffix(constants.NIFTI_EXTENSION)
                print(f"Using relabelled fallback for {method}: {seg_file}")

            if not seg_file.exists():
                print(f"Skipping {method}: file not found {seg_file}")
                continue

            seg_im = sitk.ReadImage(str(seg_file), sitk.sitkUInt8)
            seg_arr = sitk.GetArrayFromImage(seg_im).astype(np.float32)

            if reference_im is None:
                reference_im = seg_im  # save reference for geometry

            seg_arrays.append(seg_arr)

        if len(seg_arrays) == 0:
            print(f"⚠️ Skipping {patient.name}: no valid segmentations found.")
            continue

        # --- Compute voxelwise mean ---
        mean_array = np.mean(seg_arrays, axis=0)

        # --- Discretize using cutoffs ---
        discretized = np.zeros_like(mean_array, dtype=np.uint8)
        discretized[(mean_array >= 0.5) & (mean_array < 1.5)] = 1
        discretized[(mean_array >= 1.5) & (mean_array < 2.5)] = 2
        discretized[mean_array >= 2.5] = 3

        # --- Save ensemble segmentation ---
        ensemble_im = sitk.GetImageFromArray(discretized)
        ensemble_im.CopyInformation(reference_im)

        if relabelled:
            out_path = segmentations_dir.joinpath("ensemble_mean_seg_T1GD_relabeled.nii.gz")
        else:
            out_path = segmentations_dir.joinpath("ensemble_mean_seg_T1GD.nii.gz")
        sitk.WriteImage(ensemble_im, str(out_path))

        # Now we measure the Dice score of this mean ensemble segmentation with respect to the ground truth

        gt_file = gt_dir.joinpath(patient.name, "NIFTI", "MASK_original.nii.gz")
        if not gt_file.exists():
            gt_file = gt_dir.joinpath(patient.name, "NIFTI", "MASK.nii.gz")
        gt_im = sitk.ReadImage(str(gt_file), sitk.sitkUInt8)

        if not same_physical_space(ensemble_im, gt_im):
            ensemble_im.CopyInformation(gt_im)

        # per_label_dices = []

        # for label in labels:

        #     im_label = extract_label(ensemble_im, labels=label)
        #     gt_label = extract_label(gt_im, labels=label)

        #     dice_label = compute_dice_score(im_label, gt_label)
        #     per_label_dices.append(dice_label)

        # mean_dice_per_label = np.mean(per_label_dices)

        # Multiclass Dice (sum of intersections over sum of sizes)
        total_intersection = 0
        total_union = 0

        for label in labels:

            gt_bin = extract_label(gt_im, labels=label)
            pred_bin = extract_label(ensemble_im, labels=label)

            gt_arr = sitk.GetArrayFromImage(gt_bin) > 0
            pred_arr = sitk.GetArrayFromImage(pred_bin) > 0

            total_intersection += np.sum(gt_arr & pred_arr)
            total_union += np.sum(gt_arr) + np.sum(pred_arr)

        multiclass_dice = 2 * total_intersection / (total_union + 1e-6)

        print(f"✅ Saved mean ensemble segmentation for {patient.name} → {out_path}")

        # Store record for summary DataFrame
        ensemble_results.append(
            {
                "Patient": patient.name,
                "Num_Models_Used": len(seg_arrays),
                "MCDSC-GT": multiclass_dice,
                "Output_File": str(out_path),
            }
        )

    # --- Save summary table ---
    results_df = pd.DataFrame(ensemble_results)
    summary_csv = output_dir.joinpath("ensemble_mean_summary.csv")
    results_df.to_csv(summary_csv, index=False)
    print(f"\n📁 Summary saved to {summary_csv}")

    return results_df


def plot_model_vs_ensemble_correlation(
    pairwise_df: pd.DataFrame, ensemble_df: pd.DataFrame, output_dir: Path
):
    """
    Create scatter plots of per-model Mean_Dice_vs_All_Models vs ensemble WT Dice.

    One subplot per model, showing Pearson correlation with ensemble Dice.
    Saves figure and correlation table.

    Args:
        pairwise_df (pd.DataFrame): DataFrame with ['Patient', 'Method', 'Mean_Dice_vs_All_Models']
        ensemble_df (pd.DataFrame): DataFrame with ['Patient', 'MeanDSC-GT']
        output_dir (Path): Directory to save outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge data on patient name
    merged = pairwise_df.merge(ensemble_df[["Patient", "MeanDSC-GT"]], on="Patient", how="inner")

    methods = merged["Method"].unique()
    n_methods = len(methods)
    n_cols = math.ceil(math.sqrt(n_methods))
    n_rows = math.ceil(n_methods / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    corr_records = []

    for i, method in enumerate(methods):
        ax = axes[i]
        df_m = merged[merged["Method"] == method].copy()

        if len(df_m) < 3:
            print(f"⚠️ Skipping {method}: not enough patients for correlation.")
            continue

        # x = df_m["MeanDSC-GT"]
        x = df_m["MeanDSC-GT"]
        y = df_m["Mean_Dice_vs_All_Models"]

        # Compute Pearson correlation
        r, p = pearsonr(x, y)
        corr_records.append({"Method": method, "Pearson_r": r, "p_value": p})

        # Scatter plot
        ax.scatter(x, y, s=40, alpha=0.7, edgecolor="k")
        ax.set_title(f"{method}\nr={r:.3f}, p={p:.3e}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Quality score")
        ax.set_ylabel("Mean DSC vs GT")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Correlation Between Model Quality and Ensemble Dice", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    fig_path = output_dir / "model_vs_ensemble_correlation.svg"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()

    # Save correlation table
    corr_df = pd.DataFrame(corr_records)
    corr_csv = output_dir / "model_vs_ensemble_correlation.csv"
    corr_df.to_csv(corr_csv, index=False)

    print(f"\n✅ Saved correlation figure to {fig_path}")
    print(f"✅ Saved correlation table to {corr_csv}")
    print("\n📊 Pearson correlations:\n", corr_df)

    return corr_df


if __name__ == "__main__":

    input_dir = Path("/scratch/radv/share/glioseg/new_run/Patients")
    gt_dir = Path("/scratch/radv/share/glioseg/new_run/GT/")
    labels = [1, 2, 3]

    output_dir = Path("/scratch/radv/share/glioseg/new_run/QA_100_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_before_relabeling = output_dir.joinpath("before_relabeling")
    output_dir_before_relabeling.mkdir(parents=True, exist_ok=True)
    output_dir_after_relabeling = output_dir.joinpath("after_relabeling")
    output_dir_after_relabeling.mkdir(parents=True, exist_ok=True)

    segmentation_files_dict_before_relabeling = {
        "nnUNet task 001-082": constants.NNUNET_TASK001_TASK082_ENSEMBLE_SEGMENTATION_NAME,
        "nnUNet task 500": constants.NNUNET_TASK500_ENSEMBLE_SEGMENTATION_NAME,
        "HDglio": constants.HDGLIO_SEGMENTATION_NAME,
        "DeepSCAN": constants.SCAN2020_SEGMENTATION_NAME,
        "FETS": constants.FETS_SEGMENTATION_NAME,
        "MV": constants.ENSEMBLED_MAJORITY_SEGMENTATION_NAME,
        "STAPLE": constants.ENSEMBLED_SEGMENTATION_NAME,
        "SIMPLE": constants.ENSEMBLED_SIMPLE_SEGMENTATION_NAME,
    }

    segmentation_files_dict_after_relabeling = {
        "nnUNet task 001-082_relabeled": constants.NNUNET_TASK001_TASK082_ENSEMBLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
        "nnUNet task 500_relabeled": constants.NNUNET_TASK500_ENSEMBLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
        "HDglio_relabeled": constants.HDGLIO_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "DeepSCAN_relabeled": constants.SCAN2020_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "FETS_relabeled": constants.FETS_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "MV_relabeled": constants.ENSEMBLED_MAJORITY_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "STAPLE_relabeled": constants.ENSEMBLED_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "SIMPLE_relabeled": constants.ENSEMBLED_SIMPLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
    }

    # Compute mean ensemble segmentations before relabeling
    ensemble_before_relabel = compute_mean_ensemble_segmentation(
        input_dir,
        gt_dir,
        segmentation_files_dict_before_relabeling,
        labels,
        output_dir_before_relabeling,
        relabelled=False,
    )
    # Compute mean ensemble segmentations after relabeling
    ensemble_after_relabel = compute_mean_ensemble_segmentation(
        input_dir,
        gt_dir,
        segmentation_files_dict_after_relabeling,
        labels,
        output_dir_after_relabeling,
        relabelled=True,
    )

    pairwise_results_before_relabel = pd.read_csv(
        str(output_dir_before_relabeling / "quality_assessment_pairwise_mean.csv")
    )
    pairwise_results_after_relabel = pd.read_csv(
        str(output_dir_after_relabeling / "quality_assessment_pairwise_mean.csv")
    )

    plot_model_vs_ensemble_correlation(
        pairwise_results_before_relabel,
        ensemble_before_relabel,
        output_dir_before_relabeling,
    )

    plot_model_vs_ensemble_correlation(
        pairwise_results_after_relabel,
        ensemble_after_relabel,
        output_dir_after_relabeling,
    )
