"""N4ITK bias field correction while preserving NIfTI physical geometry."""

# Import numpy for numerical operations
import numpy as np

# Try importing SimpleITK, the medical imaging library that performs N4 bias correction
try:
    import SimpleITK as sitk
# Save the error if SimpleITK is not installed
except ImportError as exc:
    sitk = None
    _IMPORT_ERROR = exc
# Set error to None if import succeeds
else:
    _IMPORT_ERROR = None


# Fix uneven scanner lighting/shading (bias field) across the 3D MRI scan
def n4_correct(image: "sitk.Image", shrink_factor: int = 4,
               iterations=(50, 40, 30), convergence_threshold: float = 0.001) -> "sitk.Image":
    # Stop immediately if SimpleITK is missing
    if sitk is None:
        raise RuntimeError("N4 correction requires SimpleITK") from _IMPORT_ERROR
    # Ensure the input scan is strictly a 3D volume
    if image.GetDimension() != 3:
        raise ValueError("N4 correction expects a 3D image")

    # Cast voxel intensities to 32-bit float for numerical precision
    original = sitk.Cast(image, sitk.sitkFloat32)
    
    # Create a rough threshold mask separating head tissue from black air background
    fitting_mask = sitk.OtsuThreshold(original, 0, 1, 200)
    
    # Calculate safe downsampling factors for each dimension
    shrink = [max(1, min(int(shrink_factor), int(size))) for size in original.GetSize()]
    
    # Downsample image by factor of 4 so calculation runs in seconds instead of minutes
    small_image = sitk.Shrink(original, shrink)
    
    # Downsample the tissue mask to match the smaller image dimensions
    small_mask = sitk.Shrink(fitting_mask, shrink)
    
    # Initialize the N4 bias field correction filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    # Set multi-resolution optimization iterations from coarse to fine [50, 40, 30]
    corrector.SetMaximumNumberOfIterations([int(v) for v in iterations])
    
    # Set convergence stopping threshold (stop when field change is less than 0.001)
    corrector.SetConvergenceThreshold(float(convergence_threshold))
    
    # Calculate the smooth bias field on the downsampled image
    corrector.Execute(small_image, small_mask)
    # Reconstruct the estimated bias field at full original resolution
    log_field = corrector.GetLogBiasFieldAsImage(original)
    
    # Divide the original scan by the bias field to remove shading artifacts
    corrected = original / sitk.Exp(log_field)
    
    # Return the clean, shadow-free image as float32
    return sitk.Cast(corrected, sitk.sitkFloat32)
