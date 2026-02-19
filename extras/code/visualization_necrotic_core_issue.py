import simpleITK as sitk
import SimpleITK as sitk


def save_slice_from_nifti(input_nifti_path: str, output_nifti_path: str, slice_index: int, axis: int = 2):
    """
    Saves a slice from a 3D NIfTI volume as a 2D NIfTI file.

    Args:
        input_nifti_path (str): Path to the input NIfTI file.
        output_nifti_path (str): Path to save the output 2D slice NIfTI file.
        slice_index (int): Index of the slice to extract (based on the chosen axis).
        axis (int, optional): Axis along which to slice (default is 2, which corresponds to the z-axis).
    """
    # Read the NIfTI volume
    volume = sitk.ReadImage(input_nifti_path)
    
    # Get the size of the volume
    size = volume.GetSize()
    
    # Extract the slice along the specified axis
    if axis == 0:  # Slicing along the x-axis
        slice = volume[slice_index, :, :]
    elif axis == 1:  # Slicing along the y-axis
        slice = volume[:, slice_index, :]
    elif axis == 2:  # Slicing along the z-axis
        slice = volume[:, :, slice_index]
    else:
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z).")
    
    # Convert the slice into a 2D image
    slice_image = sitk.GetImageFromArray(slice)
    
    # Save the slice as a new NIfTI file
    sitk.WriteImage(slice_image, output_nifti_path)
    print(f"Slice saved to {output_nifti_path}")

