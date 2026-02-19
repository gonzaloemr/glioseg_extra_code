import os

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


def same_physical_space(img1: sitk.Image, img2: sitk.Image, verbose: bool = False) -> bool:
    """
    Check if two SimpleITK images share the same physical space 
    (size, spacing, origin, direction).

    Args:
        img1 (sitk.Image): First image.
        img2 (sitk.Image): Second image.
        verbose (bool): If True, prints differences.

    Returns:
        bool: True if all spatial properties match, False otherwise.
    """
    same_size = img1.GetSize() == img2.GetSize()
    same_spacing = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetSpacing(), img2.GetSpacing()))
    same_origin = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetOrigin(), img2.GetOrigin()))
    same_direction = all(abs(a - b) < 1e-6 for a, b in zip(img1.GetDirection(), img2.GetDirection()))

    if verbose:
        print(f"Size match:       {same_size} ({img1.GetSize()} vs {img2.GetSize()})")
        print(f"Spacing match:    {same_spacing} ({img1.GetSpacing()} vs {img2.GetSpacing()})")
        print(f"Origin match:     {same_origin} ({img1.GetOrigin()} vs {img2.GetOrigin()})")
        print(f"Direction match:  {same_direction}")

    return all([same_size, same_spacing, same_origin, same_direction])


def extract_label(mask: sitk.Image, labels: int | list[int]) -> sitk.Image:
    """
    Binarizes a segmentation mask by thresholding specific labels.

    Args:
        mask (sitk.Image): The input segmentation msask as a SimpleITK image.
        labels (int | list[int]): The label value(s) to be binarized.
            - If a single integer, voxels with this value will be set to 1, and all others to 0.
            - If a list, voxels with any of these values will be set to 1, and all others to 0.

    Returns:
        sitk.Image: A binarized mask where the specified label(s) are set to 1, and all other values are 0.
    """
    if isinstance(labels, int):
        return sitk.BinaryThreshold(mask, lowerThreshold=labels, upperThreshold=labels)
    binary_mask = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
    binary_mask.CopyInformation(mask)
    for label in labels:
        binary_mask = binary_mask | sitk.BinaryThreshold(
            mask, lowerThreshold=label, upperThreshold=label
        )
    return binary_mask


def compute_dice_score(
    gt_im: sitk.Image, seg_im: sitk.Image,
) -> float:
    
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()

    gt_sum = np.sum(sitk.GetArrayFromImage(gt_im))
    pred_sum = np.sum(sitk.GetArrayFromImage(seg_im))

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    else:
        overlap_filter.Execute(gt_im, seg_im)
        return overlap_filter.GetDiceCoefficient()
    
def compute_multiclass_dice_score(
    gt_im: sitk.Image, seg_im: sitk.Image, label_ids: list[int],
) -> float:
    
    total_intersection = 0
    total_union = 0

    for label in label_ids:

        gt_mask = extract_label(gt_im, labels=label)
        pred_mask = extract_label(seg_im, labels=label)

        gt_arr = sitk.GetArrayFromImage(gt_mask)
        pred_arr = sitk.GetArrayFromImage(pred_mask)

        total_intersection += np.sum(gt_arr & pred_arr)
        total_union += np.sum(gt_arr) + np.sum(pred_arr)

    multi_class_dice = (2 * total_intersection) / (total_union + 1e-6)
    
    return multi_class_dice


def compute_metrics(pred_root: str, gt_root: str) -> pd.DataFrame:
    """
    Compute Dice scores for all folds in the given directory structure.

    Args:
        pred_root (str): Path to the directory containing fold_0, fold_1, ..., fold_4.
        gt_root (str): Path to the ground truth directory (patient_id/MASK.nii.gz).

    Returns:
        pd.DataFrame: Columns = [patient_id, dice_score], one row per test case.
    """
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)

    labels = [1,2,3]

    all_results = []
    for fold_dir in sorted(pred_root.glob("fold_*")):
        print("hola in the loop")
        print(fold_dir)
        val_dir = fold_dir / "validation_atlas"
        if not val_dir.exists():
            print(f"Skipping {fold_dir} (no validation_atlas folder)")
            continue

        # Get patient IDs from all *_T1GD.nii.gz files
        pred_files = list(val_dir.glob("*_T1GD.nii.gz"))
        patient_ids = [f.name.replace("_T1GD.nii.gz", "") for f in pred_files]

        # Compute Dice per patient
        for pid in patient_ids:
            gt_file = gt_root / pid / "NIFTI" / "MASK_original.nii.gz"
            
            if not gt_file.exists():
                gt_file = gt_root / pid / "NIFTI" / "MASK.nii.gz"
            pred_file = val_dir / f"{pid}_T1GD.nii.gz"

            if not gt_file.exists():
                print(f"Missing GT for {pid}, skipping.")
                continue
            if not pred_file.exists():
                print(f"Missing prediction for {pid}, skipping.")
                continue

            # Read images

            gt = sitk.ReadImage(str(gt_file), sitk.sitkUInt8)
            pred = sitk.ReadImage(str(pred_file), sitk.sitkUInt8)

            if not same_physical_space(gt, pred):
                pred.CopyInformation(gt)

            gt_wt = extract_label(gt, labels)
            pred_wt = extract_label(pred, labels)

            dice_wt = compute_dice_score(gt_wt, pred_wt)
            multiclass_dice = compute_multiclass_dice_score(gt, pred, labels)

            all_results.append({"Case": pid, "WT-DSC": dice_wt, "MC-DSC": multiclass_dice})

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    # Add mean and standard deviation as two last rows 
    mean_row = pd.DataFrame(
        {
            "Case": ["Mean"],
            "WT-DSC": [df["WT-DSC"].mean()],
            "MC-DSC": [df["MC-DSC"].mean()],
        }
    )
    std_row = pd.DataFrame(
        {
            "Case": ["Std"],
            "WT-DSC": [df["WT-DSC"].std()],
            "MC-DSC": [df["MC-DSC"].std()],
        }
    )
    df = pd.concat([df, mean_row, std_row], ignore_index=True)
    print(f"\nComputed Dice for {len(df)} patients.\n")

    return df

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metrics_boxplot(df: pd.DataFrame, save_path: str = None):
    """
    Plot boxplots for WT-DSC and MC-DSC with distinct colors, mean (diamond marker),
    and mean ± std shown below each box.

    Args:
        df (pd.DataFrame): DataFrame returned by compute_metrics()
                           containing 'WT-DSC' and 'MC-DSC' columns.
        save_path (str, optional): Path to save the plot image.
    """
    # Filter out summary rows
    df = df[~df["Case"].isin(["Mean", "Std"])].copy()

    metrics = ["WT-DSC", "MC-DSC"]
    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange

    # Compute mean ± std
    means = df[metrics].mean()
    stds = df[metrics].std()

    # Melt for plotting
    df_melted = df.melt(id_vars="Case", value_vars=metrics, var_name="Metric", value_name="Dice Score")

    # Plot setup
    plt.figure(figsize=(8, 6))
    sns.set(style="ticks")

    ax = sns.boxplot(
        x="Metric", y="Dice Score", hue="Metric", data=df_melted, showmeans=False, legend=True
    )


    # Plot mean diamonds
    for i, metric in enumerate(metrics):
        plt.scatter(
            i, means[metric], 
            marker="D", s=30, edgecolors="black", facecolors="none", zorder=10
        )

    # # Annotate mean ± std
    # for i, metric in enumerate(metrics):
    #     mean_val, std_val = means[metric], stds[metric]
    #     plt.text(
    #         i, -0.08, f"{mean_val:.3f} ± {std_val:.3f}",
    #         ha="center", va="center", fontsize=11, color="black", transform=ax.get_xaxis_transform()
    #     )

    # Replace x-ticks with mean ± std
    new_labels = [f"{means[m]:.3f} ± {stds[m]:.3f}" for m in metrics]
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(new_labels, fontsize=12)

    # Aesthetics
    ax.set_title("Segmentation peformance nnUNet 2D")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove gridlines and unnecessary borders
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D


def plot_MCDSC_boxplot(df_2d: pd.DataFrame, df_3d: pd.DataFrame, save_path: str = None):
    """
    Create a single boxplot comparing MC-DSC distributions for 2D and 3D models,
    with legend for colors and mean marker.

    Args:
        df_2d (pd.DataFrame): DataFrame containing 'MC-DSC' for 2D model.
        df_3d (pd.DataFrame): DataFrame containing 'MC-DSC' for 3D model.
        save_path (str, optional): Path to save the figure.
    Returns:
        fig, ax
    """

    # Remove summary rows if present
    df_2d = df_2d[~df_2d["Case"].isin(["Mean", "Std"])].copy()
    df_3d = df_3d[~df_3d["Case"].isin(["Mean", "Std"])].copy()

    # Prepare data for plotting
    df_2d_plot = df_2d[["MC-DSC"]].copy()
    df_2d_plot["Model"] = "2D"

    df_3d_plot = df_3d[["MC-DSC"]].copy()
    df_3d_plot["Model"] = "3D"

    df_plot = pd.concat([df_2d_plot, df_3d_plot], ignore_index=True)

    # Compute stats
    means = df_plot.groupby("Model")["MC-DSC"].mean()
    stds = df_plot.groupby("Model")["MC-DSC"].std()

    # Colors (matching the boxplot palette)
    colors = {"2D": "#1f77b4", "3D": "#ff7f0e"}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(style="ticks")

    ax = sns.boxplot(
        x="Model",
        y="MC-DSC",
        data=df_plot,
        palette=[colors["2D"], colors["3D"]],
        showmeans=False,
        ax=ax
    )

    # Plot mean diamonds (use ax to keep everything on same axes)
    for i, model in enumerate(["2D", "3D"]):
        ax.scatter(
            i,
            means[model],
            marker="D",
            s=60,
            edgecolors="black",
            facecolors="white",
            zorder=10,
            label="_nolegend_",  # avoid auto legend entries here
        )

    # X-tick labels as mean ± std
    xtick_labels = [f"{means[m]:.3f} ± {stds[m]:.3f}" for m in ["2D", "3D"]]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xtick_labels, fontsize=12)

    ax.set_title("MC-DSC Comparison: 2D vs 3D nnUNet", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("MC-DSC")

    ax.grid(False)

    # --- Build legend manually ---
    color_patches = [
        mpatches.Patch(color=colors["2D"], label="2D"),
        mpatches.Patch(color=colors["3D"], label="3D"),
    ]
    mean_marker = Line2D([0], [0], marker="D", color="w", label="Mean",
                         markerfacecolor="white", markeredgecolor="black", markersize=8)

    # Place legend (adjust location as you like)
    handles = color_patches + [mean_marker]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved boxplot to {save_path}")

    return fig, ax


if __name__ == "__main__":

    from pathlib import Path

    # Example usage

    predictions_dir = "/scratch/radv/gemosquerarojas/nnUNet_results/Dataset103_Glioseg_new_GT/nnUNetTrainer__nnUNetPlans__2d/"

    # predictions_dir_2d = "/scratch/radv/gemosquerarojas/nnUNet_results/Dataset103_Glioseg_new_GT/nnUNetTrainer__nnUNetPlans__2d/"  # contains fold_0/.../validation_atlas/
    # predictions_dir_3d = "/scratch/radv/gemosquerarojas/nnUNet_results/Dataset103_Glioseg_new_GT/nnUNetTrainer__nnUNetPlans__3d_fullres/"  # contains fold_0/.../validation_atlas/
    ground_truth_dir = "/scratch/radv/share/glioseg/new_run_corrected/GT/"  # contains patient_id/MASK.nii.gz

    df = compute_metrics(predictions_dir, ground_truth_dir)
    # df_2d = compute_metrics(predictions_dir_2d, ground_truth_dir)s
    # df_3d = compute_metrics(predictions_dir_3d, ground_truth_dir)

    # Optionally save
    df.to_csv(Path(predictions_dir).joinpath("metrics.csv"), index=False)

    # Plot boxplot
    plot_metrics_boxplot(df, save_path=Path(predictions_dir).joinpath("metrics_boxplot.svg"))
    plot_metrics_boxplot(df, save_path=Path(predictions_dir).joinpath("metrics_boxplot.png"))

    # plot_MCDSC_boxplot(
    #     df_2d,
    #     df_3d,
    #     save_path=Path(predictions_dir_2d).parent.joinpath("MCDSC_2D_vs_3D_boxplot.svg"),
    # )
