import itertools
import math

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy.stats import pearsonr
from sklearn.metrics import auc

import glioseg.constants as constants

from glioseg.segmentation.tumor.validation.metrics_nnunet import compute_dice_score
from glioseg.segmentation.tumor.validation.metrics_nnunet import extract_label
from glioseg.segmentation.tumor.validation.metrics_nnunet import same_physical_space


def compute_quality_assessment_pairwise_mean(input_dir, gt_dir, segmentation_dict, labels, output_dir):
    """
    Compute per-model quality score:
    For each model, average its mean per-label Dice score against all other models.
    """

    all_results = []

    for patient in input_dir.iterdir():
        
        segmentations_dir = patient.joinpath("SEGMENTATIONS", "NIFTI")
        gt_file = gt_dir.joinpath(patient.name, "NIFTI", "MASK_original.nii.gz")
        
        if not gt_file.exists():
            gt_file = gt_dir.joinpath(patient.name, "NIFTI", "MASK.nii.gz")

        gt_im = sitk.ReadImage(str(gt_file), sitk.sitkUInt8)

        # --- Load all model segmentations ---
        seg_images = {}
        for method, seg_file_name in segmentation_dict.items():
            seg_file = segmentations_dir.joinpath(constants.TUMOR_MASK_NAME+seg_file_name+"_T1GD").with_suffix(constants.NIFTI_EXTENSION)
            
            if not seg_file.exists() and "_relabelled" in seg_file_name:
                seg_file_name = seg_file_name.replace("_relabelled", "")
                print(f"Segmentation file {seg_file} does not exist, and its relabelled")
                print(f"Using instead {seg_file_name}")
                seg_file = segmentations_dir.joinpath(constants.TUMOR_MASK_NAME+seg_file_name+"_T1GD").with_suffix(constants.NIFTI_EXTENSION)

            if seg_file.exists():
                seg_images[method] = sitk.ReadImage(str(seg_file), sitk.sitkUInt8)
            else:
                print(f"Skipping {method}: file not found {seg_file}")

        methods = list(seg_images.keys())
        if len(methods) < 2:
            print(f"Skipping {patient.name}: less than 2 segmentations.")
            continue

        # --- Initialize per-model accumulators ---
        dice_sums = {m: 0.0 for m in methods}
        dice_counts = {m: 0 for m in methods}

        dice_sums_multiclass = {m: 0.0 for m in methods}
        dice_counts_multiclass = {m: 0 for m in methods}

        # --- Pairwise comparisons ---


        for m1, m2 in itertools.combinations(methods, 2):
            seg1, seg2 = seg_images[m1], seg_images[m2]

            if not same_physical_space(seg1, seg2):
                seg2.CopyInformation(seg1)

            # Dice per label → mean per pair
            per_label_dices = []

            total_intersection_pair = 0
            total_union_pair = 0

            for label in labels:
                
                s1_label = extract_label(seg1, label)
                s2_label = extract_label(seg2, label)
                dice = compute_dice_score(s1_label, s2_label)

                s1_arr = sitk.GetArrayFromImage(s1_label) > 0
                s2_arr = sitk.GetArrayFromImage(s2_label) > 0

                total_intersection_pair += np.sum(s1_arr & s2_arr)
                total_union_pair += np.sum(s1_arr) + np.sum(s2_arr)


                per_label_dices.append(dice)
            
            mean_dice_pair = np.mean(per_label_dices)
            multiclass_dice_pair = 2 * total_intersection_pair / (total_union_pair + 1e-6)

            # Add contribution to both models
            dice_sums[m1] += mean_dice_pair
            dice_sums[m2] += mean_dice_pair
            dice_counts[m1] += 1
            dice_counts[m2] += 1

            # Add contribution to both models (multiclass)
            dice_sums_multiclass[m1] += multiclass_dice_pair
            dice_sums_multiclass[m2] += multiclass_dice_pair
            dice_counts_multiclass[m1] += 1
            dice_counts_multiclass[m2] += 1

        # --- Compute per-model mean dice (vs all others) and Dice score with the ground truth---
        for method in methods:
            
            if dice_counts[method] == 0:
                continue
            
            confidence_dice_mircros = dice_sums[method] / dice_counts[method]
            confidence_dice_micro = dice_sums_multiclass[method] / dice_counts_multiclass[method]


            seg_im = seg_images[method]

            if not same_physical_space(gt_im, seg_im):
                seg_im.CopyInformation(gt_im)

            dice_scores_labels = []

            total_intersection = 0
            total_union = 0

            for label in labels: 
                
                gt_label = extract_label(gt_im, label)
                seg_label = extract_label(seg_im, label)

                gt_arr = sitk.GetArrayFromImage(gt_label) > 0
                seg_arr = sitk.GetArrayFromImage(seg_label) > 0

                total_intersection += np.sum(gt_arr & seg_arr)
                total_union += np.sum(gt_arr) + np.sum(seg_arr)

                dice_score_per_label = compute_dice_score(gt_label, seg_label)
                dice_scores_labels.append(dice_score_per_label)
            
            mean_dice_score_vs_gt = np.mean(dice_scores_labels)
            multiclass_dice = 2 * total_intersection / (total_union + 1e-6)
            
            all_results.append({
                "Patient": patient.name,
                "Method": method,
                "PW-DSC": confidence_dice_mircros,
                "PW-MCDSC": confidence_dice_micro,
                "GT-DSC": mean_dice_score_vs_gt,
                "GT-MCDSC": multiclass_dice
            })

    # --- Save results ---
    results = pd.DataFrame(all_results)
    output_file = output_dir.joinpath("QA_results_summary.csv")
    results.to_csv(output_file, index=False)
    print(f"Quality assessment results saved to {output_file}")

    return results

def plot_quality_vs_risk(
    results_df: pd.DataFrame,
    model_keys: str | list[str] | None,
    output_dir: Path, 
    metric_compute: str,
):
    """
    Plot PW-DSC (x-axis) vs Risk (1 - GT-DSC) (y-axis) for one or more models.

    Args:
        results_df (pd.DataFrame): Must contain ["Patient", "Method", "PW-DSC", "GT-DSC"].
        model_keys (str | list[str] | None): Model(s) to plot.
            - str → single model plot
            - list[str] → multiple models, one subplot each
            - None → all models found in results_df             
        output_dir (Path): Directory to save outputs.
        metric_compute (str): Type of metric either "macro" or "micro". If "macro", it corresponds to the mean per-label Dice score, 
        if "micro", it corresponds to the multiclass Dice score.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine models to plot ---
    if model_keys is None:
        models_to_plot = results_df["Method"].unique()
    elif isinstance(model_keys, str):
        models_to_plot = [model_keys]
    elif isinstance(model_keys, list):
        models_to_plot = model_keys
    else:
        raise ValueError("model_keys must be None, a string, or a list of strings.")
    
    assert metric_compute in ["macro", "micro"], "metric_compute must be either 'macro' or 'micro'"

    if metric_compute == "macro":
        GT_key = "GT-DSC"
        confidence_key = "PW-DSC"
    else:
        GT_key = "GT-MCDSC"
        confidence_key = "PW-MCDSC"

    n_models = len(models_to_plot)

    # --- Handle single vs multiple models ---
    if n_models == 1:
        
        model = models_to_plot[0]
        df = results_df[results_df["Method"] == model].copy()
        if df.empty:
            print(f"No entries found for model '{model}'")
            return

        df["Risk"] = 1 - df[GT_key]
        x, y = df[confidence_key].values, df["Risk"].values

        r_value, p_value = pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, color="tab:blue", edgecolor="k", s=70, alpha=0.8)
        plt.plot(x, slope * x + intercept, color="red", linestyle="--", linewidth=2, label="Linear fit")
        plt.title(f"{model}\nPearson r={r_value:.3f}, p={p_value:.3e}", fontsize=11, fontweight="bold")
        plt.xlabel(f"Confidence ({confidence_key})")
        plt.ylabel(f"Risk (1- {GT_key})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        output_file = output_dir / f"scatter_QA_vs_Risk_{metric_compute}.svg"

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved scatter plot for model '{model}' → {output_file}")
        print(f"   Pearson r = {r_value:.3f}, p = {p_value:.3e}")

        return {"Model": model, "Pearson_r": r_value, "p_value": p_value, "Output_File": str(output_file)}

    else:
        # --- Multi-model figure ---

        if n_models <= 3:
            n_rows = 1
            n_cols = n_models
        else:
            n_cols = math.ceil(math.sqrt(n_models))
            n_rows = math.ceil(n_models / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()

        correlation_records = []

        for i, model in enumerate(models_to_plot):
            ax = axes[i]
            df = results_df[results_df["Method"] == model].copy()
            if df.empty:
                ax.set_visible(False)
                continue
            
            df["Risk"] = 1 - df[GT_key]
            x, y = df[confidence_key].values, df["Risk"].values

            r_value, p_value = pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)

            ax.scatter(x, y, color=f"C{i % 10}", edgecolor="k", s=70, alpha=0.8)
            ax.plot(x, slope * x + intercept, color="red", linestyle="--", linewidth=2)
            ax.set_title(f"{model}\nr={r_value:.3f}, p={p_value:.3e}", fontsize=10, fontweight="bold")
            ax.set_xlabel(f"Confidence ({confidence_key})")
            ax.set_ylabel(f"Risk (1 - {GT_key})")
            ax.grid(True, linestyle="--", alpha=0.5)

            correlation_records.append({"Model": model, "Pearson_r": r_value, "p_value": p_value})

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # fig.suptitle("PW-DSC vs Risk (1 - GT-DSC) per model", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        output_file = output_dir / f"scatter_QA_vs_Risk_{metric_compute}.svg"
        plt.savefig(output_file, dpi=300)
        plt.close()

        corr_df = pd.DataFrame(correlation_records)
        corr_csv  = output_dir / f"correlation_summary_{metric_compute}.csv"
        corr_df.to_csv(corr_csv, index=False)

        print(f"Saved multi-model figure {output_file}")
        print(f"Correlation summary saved {corr_csv}")

        return corr_df

import math

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import auc


def plot_risk_coverage_subplots(results: pd.DataFrame, output_dir: Path, model_keys, metric_compute: str):
    
    """
    Generate risk–coverage curves for one, several, or all models.
    Each subplot shows:
        - Empirical risk–coverage curve (PW-DSC vs GT-DSC)
        - Optimal curve (best possible ordering by GT-DSC)
        - AUC of empirical curve

    Args:
        results (pd.DataFrame): Must contain ['Patient', 'Method', 'PW-DSC', 'GT-DSC'].
        output_dir (Path): Directory to save outputs.
        model_keys (str | list[str] | None): Which models to plot. None = all.
        metric_compute (str): Type of GT Dice to use, either "macro" or "micro". If "macro", it corresponds to the mean per-label Dice score,
        if "micro", it corresponds to the multiclass Dice score.

    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if model_keys is None:
        methods = results["Method"].unique()
    elif isinstance(model_keys, str):
        methods = [model_keys]
    else:
        methods = model_keys

    assert metric_compute in ["macro", "micro"], "metric_compute must be either 'macro' or 'micro'"

    if metric_compute == "macro":
        GT_key = "GT-DSC"
        confidence_key = "PW-DSC"
    else:
        GT_key = "GT-MCDSC"
        confidence_key = "PW-MCDSC"

    n_methods = len(methods)

    if n_methods <= 3:
        n_rows = 1
        n_cols = n_methods
    else:
        n_cols = math.ceil(math.sqrt(n_methods))
        n_rows = math.ceil(n_methods / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    auc_results, detailed_results = [], []

    for i, method in enumerate(methods):
        
        ax = axes[i]
        df = results[results["Method"] == method].copy()
        total_cases = len(df)
        thresholds = np.linspace(0, 1, total_cases)  # threshold sweep
        if total_cases == 0:
            continue

        sorted_risks = 1 - np.sort(df[GT_key].values)[::-1]

        coverages, avg_risks, optimal_risks = [], [], []

        for t in thresholds:
            # Empirical curve
            subset = df[df[confidence_key] >= t]
            coverage = len(subset) / total_cases
            avg_risk = np.mean(1 - subset[GT_key]) if len(subset) > 0 else 0.0

            # Optimal curve: take top-k by ordered risk (same k as empirical coverage)
            k = len(subset)
            optimal_risk = np.mean(sorted_risks[:k]) if k > 0 else 0.0

            coverages.append(coverage)
            avg_risks.append(avg_risk)
            optimal_risks.append(optimal_risk)

            # Save detailed threshold info
            detailed_results.append({
                "Method": method,
                "Threshold": round(t, 3),
                "Coverage": coverage,
                "Average_Risk": avg_risk,
                "Optimal_Risk": optimal_risk
            })

        coverages = np.array(coverages)
        avg_risks = np.array(avg_risks)
        optimal_risks = np.array(optimal_risks)

        auc_value = auc(coverages, avg_risks)
        optimal_auc = auc(coverages, optimal_risks)


        area_between = np.abs(auc_value - optimal_auc)

        score = np.mean([auc_value,area_between])

        auc_results.append({"Method": method, "AUC": auc_value, "AUC_Optimal": optimal_auc, "Area_Between": area_between, "Score": score})

        # Plot curves
        # ax.plot(coverages, avg_risks, "o-", lw=2, color=f"C{i%10}", label="Empirical")
        ax.plot(coverages, avg_risks, lw=2, color=f"C{i%10}")
        ax.fill_between(coverages, avg_risks, alpha=0.2, color=f"C{i%10}", label=f"AUC {auc_value:.3f}")  # shaded area
        ax.plot(coverages, optimal_risks, "--", lw=2, color="black", alpha=0.6, label="Optimal")
        ax.set_title(f"{method}\n GAP-AUC={area_between:.3f} \n Score={score:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coverage")
        ax.set_ylabel(f"Average Risk (1 - {GT_key})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend()

    # Remove extra axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # fig.suptitle("Risk–Coverage Curves per Method", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    fig_path = output_dir / f"risk_coverage_subplots_{metric_compute}.svg"
    plt.savefig(fig_path)

    # Save AUC table
    auc_df = pd.DataFrame(auc_results)
    auc_df_path = output_dir / f"risk_coverage_auc_{metric_compute}.csv"
    auc_df.to_csv(auc_df_path, float_format="%.3f", index=False)

    # Save detailed threshold results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df_path = output_dir / f"risk_coverage_detailed_{metric_compute}.csv"
    detailed_df.to_csv(detailed_df_path, float_format="%.3f", index=False)

    print(f"Saved figure to {fig_path}")
    print(f"Saved AUC table to {auc_df_path}")
    print(f"Saved detailed threshold data to {detailed_df_path}")

    return auc_df, detailed_df
 

if __name__ == "__main__":


    segmentation_files_dict_before_relabeling = {
    "nnUNet": constants.NNUNET_TASK001_TASK082_ENSEMBLE_SEGMENTATION_NAME,
    "BraTS21": constants.NNUNET_TASK500_ENSEMBLE_SEGMENTATION_NAME,
    "HDglio": constants.HDGLIO_SEGMENTATION_NAME,
    "DeepSCAN": constants.SCAN2020_SEGMENTATION_NAME,
    "FETS": constants.FETS_SEGMENTATION_NAME,
    "MV": constants.ENSEMBLED_MAJORITY_SEGMENTATION_NAME,
    "STAPLE": constants.ENSEMBLED_SEGMENTATION_NAME,
    "SIMPLE": constants.ENSEMBLED_SIMPLE_SEGMENTATION_NAME
    }

    segmentation_files_dict_after_relabeling = {
        "nnUNet": constants.NNUNET_TASK001_TASK082_ENSEMBLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
        "BraTS21": constants.NNUNET_TASK500_ENSEMBLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
        "HDglio": constants.HDGLIO_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "DeepSCAN": constants.SCAN2020_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "FETS": constants.FETS_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "MV": constants.ENSEMBLED_MAJORITY_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "STAPLE": constants.ENSEMBLED_SEGMENTATION_NAME + constants.RELABELLED_NAME,
        "SIMPLE": constants.ENSEMBLED_SIMPLE_SEGMENTATION_NAME
        + constants.RELABELLED_NAME,
    }


    input_dir = Path("/scratch/radv/share/glioseg/new_run/Patients")
    gt_dir = Path("/scratch/radv/share/glioseg/new_run/GT/")
    labels = [1, 2, 3]

    output_dir = Path("/scratch/radv/share/glioseg/new_run/QA_mircros_micro")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_before_relabeling = output_dir.joinpath("before_relabeling")
    output_dir_before_relabeling.mkdir(parents=True, exist_ok=True)
    output_dir_after_relabeling = output_dir.joinpath("after_relabeling")
    output_dir_after_relabeling.mkdir(parents=True, exist_ok=True)

    output_dir_before_relabeling_all_models = output_dir_before_relabeling.joinpath("all_models")
    output_dir_before_relabeling_all_models.mkdir(parents=True, exist_ok=True)

    output_dir_after_relabeling_all_models = output_dir_after_relabeling.joinpath("all_models")
    output_dir_after_relabeling_all_models.mkdir(parents=True, exist_ok=True)

    output_dir_before_relabeling_only_ensemble = output_dir_before_relabeling.joinpath("only_ensemble_models")
    output_dir_before_relabeling_only_ensemble.mkdir(parents=True, exist_ok=True)

    output_dir_after_relabeling_only_ensemble = output_dir_after_relabeling.joinpath("only_ensemble_models")
    output_dir_after_relabeling_only_ensemble.mkdir(parents=True, exist_ok=True)

    only_ensemble = ["MV", "STAPLE", "SIMPLE"]


    # quality_before_relabeling = compute_quality_assessment_pairwise_mean(input_dir, gt_dir, segmentation_files_dict_before_relabeling, labels, output_dir_before_relabeling)
    # quality_after_relabeling = compute_quality_assessment_pairwise_mean(input_dir, gt_dir, segmentation_files_dict_after_relabeling, labels, output_dir_after_relabeling)

    quality_before_relabeling = pd.read_csv(output_dir_before_relabeling.joinpath("QA_results_summary.csv"))
    quality_after_relabeling = pd.read_csv(output_dir_after_relabeling.joinpath("QA_results_summary.csv"))

    for metric_compute in ["macro", "micro"]:
    
        plot_quality_vs_risk(
            results_df= quality_before_relabeling,
            model_keys=None,
            output_dir=output_dir_before_relabeling_all_models,
            metric_compute = metric_compute
        )
        
        plot_quality_vs_risk(
            results_df= quality_after_relabeling,
            model_keys=None,
            output_dir=output_dir_after_relabeling_all_models,
            metric_compute = metric_compute
        )

        plot_risk_coverage_subplots(
            results= quality_before_relabeling,
            output_dir=output_dir_before_relabeling_all_models,
            model_keys=None,
            metric_compute = metric_compute
        )
        
        plot_risk_coverage_subplots(
            results= quality_after_relabeling,
            output_dir=output_dir_after_relabeling_all_models,
            model_keys=None,
            metric_compute = metric_compute
        )

        
        plot_quality_vs_risk(
            results_df= quality_before_relabeling,
            model_keys=only_ensemble,
            output_dir=output_dir_before_relabeling_only_ensemble,
            metric_compute = metric_compute
        )
        
        plot_quality_vs_risk(
            results_df= quality_after_relabeling,
            model_keys=only_ensemble,
            output_dir=output_dir_after_relabeling_only_ensemble,
            metric_compute = metric_compute
        )

        plot_risk_coverage_subplots(
            results= quality_before_relabeling,
            output_dir=output_dir_before_relabeling_only_ensemble,
            model_keys=only_ensemble,
            metric_compute = metric_compute
        )
        
        plot_risk_coverage_subplots(
            results= quality_after_relabeling,
            output_dir=output_dir_after_relabeling_only_ensemble,
            model_keys=only_ensemble,
            metric_compute = metric_compute
        )