from itertools import compress

import numpy as np
import pandas as pd


def save_metrics_to_excel(data_array, models, patients, metrics, labels, filename="metrics.xlsx"):
    """
    Saves structured metrics tables to an Excel file, with each metric having its own sheet.
    
    Args:
        data_array (np.ndarray): A NumPy array of shape (n_patients, n_labels, n_models, n_metrics)
            containing the metric values. NaN values indicate missing data.
        models (list of str): List of model names.
        patients (list of str): List of patient names.
        metrics (list of str): List of metric names (each metric gets a separate sheet).
        labels (list of str): List of structure names (e.g., "Liver", "Necrosis").
        filename (str, optional): Name of the output Excel file. Defaults to "metrics.xlsx".
    """
    n_patients, n_labels, n_models, n_metrics = data_array.shape
    expected_shape = (len(patients), len(labels), len(models), len(metrics))

    # Ensure input shape matches expectations
    if data_array.shape != expected_shape:
        raise ValueError(f"Expected data_array to have shape {expected_shape}, but got {data_array.shape}")

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        workbook = writer.book

        for metric_idx, metric_name in enumerate(metrics):
            sheet_name = metric_name[:31]  # Excel sheet names max length is 31 characters

            # Extract data for this metric (shape: n_patients, n_labels, n_models)
            metric_data = data_array[:, :, :, metric_idx]

            # Filter out models that contain only NaNs for a given label
            valid_models_per_label = [
                [not np.all(np.isnan(metric_data[:, label_idx, model_idx])) for model_idx in range(n_models)]
                for label_idx in range(n_labels)
            ]

            # Filtered headers (removing models that do not segment a given label)
            filtered_models = [
                list(compress(models, valid_models_per_label[label_idx])) for label_idx in range(n_labels)
            ]

            # Generate headers dynamically
            header1 = ["Patient"] + [metric_name] + [""] * (sum(map(len, filtered_models)) - 1)  # Metric row
            header2 = ["Patient"] + [label for label, valid_models in zip(labels, filtered_models) for _ in valid_models]  # Labels row
            header3 = ["Patient"] + [model for valid_models in filtered_models for model in valid_models]  # Models row

            # Process data dynamically
            data_list = []
            for i in range(n_patients):
                row_data = [patients[i]]
                for label_idx, valid_models in enumerate(valid_models_per_label):
                    row_data.extend(metric_data[i, label_idx, valid_models].flatten())
                data_list.append(row_data)

            # Compute mean and standard deviation for valid models
            mean_values = ["Mean"]
            std_values = ["Std"]
            for label_idx, valid_models in enumerate(valid_models_per_label):
                mean_values.extend(np.round(np.nanmean(metric_data[:, label_idx, valid_models], axis=0), 3))
                std_values.extend(np.round(np.nanstd(metric_data[:, label_idx, valid_models], axis=0), 3))

            data_list.append(mean_values)
            data_list.append(std_values)

            # Convert to DataFrame
            df = pd.DataFrame([header1, header2, header3] + data_list)

            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            worksheet = writer.sheets[sheet_name]

            # Define formats
            bold_centered_format = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
            centered_format = workbook.add_format({"align": "center", "valign": "vcenter", "border": 1})
            bold_format = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})

            # Merge "Patient" from row 0 to 2
            worksheet.merge_range(0, 0, 2, 0, "Patient", bold_centered_format)

            # Merge metric name across all label columns
            worksheet.merge_range(0, 1, 0, len(header1) - 1, metric_name, bold_centered_format)

            # Merge each label name across all its valid model columns
            col_start = 1
            for label, valid_models in zip(labels, filtered_models):
                col_end = col_start + len(valid_models) - 1
                worksheet.merge_range(1, col_start, 1, col_end, label, bold_centered_format)
                col_start = col_end + 1

            # Apply formatting to headers
            for row in range(3):
                for col in range(len(df.columns)):
                    worksheet.write(row, col, df.iloc[row, col], bold_centered_format)

            # Apply formatting to data cells
            for row in range(3, len(df) - 2):  # Exclude last two rows (Mean & Std)
                for col in range(len(df.columns)):
                    worksheet.write(row, col, df.iloc[row, col], centered_format)

            # Format "Mean" and "Std" rows
            for col in range(len(df.columns)):
                worksheet.write(len(df) - 2, col, df.iloc[len(df) - 2, col], bold_format)
                worksheet.write(len(df) - 1, col, df.iloc[len(df) - 1, col], bold_format)

if __name__ == "__main__":
    # Define models, labels, and patients
    models = ["Model A", "Model B", "Model C"]
    labels = ["Label A", "Label B", "Label C"]
    patients = ["Patient 1", "Patient 2", "Patient 3"]
    metrics = ["Dice Score", "Hausdorff Distance"]

    # Create a mock dataset (3 patients, 3 labels, 3 models, 2 metrics)
    data_array = np.random.rand(3, 3, 3, 2)  # Random values

    # Set "Model B" for "Label A" to NaN
    data_array[:, 0, 1, :] = np.nan

    # Save metrics to Excel
    save_metrics_to_excel(data_array, models, patients, metrics, labels, filename="metrics_je.xlsx")
