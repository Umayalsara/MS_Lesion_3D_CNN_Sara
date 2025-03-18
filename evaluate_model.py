import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from data_preparation import load_image
from combined_script import extract_patches_from_image

# Disable XLA compilation
tf.config.optimizer.set_jit(False)
print("XLA compilation disabled.")

# Load the best model
best_model_path = 'best_model_tuned_trial_0.keras'  # Change this to the path of the best model
best_model = tf.keras.models.load_model(best_model_path)

# Define file paths for validation data
new_image_path = '/nesi/project/uoa04272/software/tensorflow-2.17.0/MS_DETECTION_3D_CNN/Test_Detect_lesion/14_test_flair_BIAScorrected.nii.gz'
output_path = '/nesi/project/uoa04272/software/tensorflow-2.17.0/MS_DETECTION_3D_CNN/Test_Detect_lesion/14_test_flair_BIAScorrected_predicted_lesions.nii'
highlighted_output_path = '/nesi/project/uoa04272/software/tensorflow-2.17.0/MS_DETECTION_3D_CNN/Test_Detect_lesion/14_test_flair_BIAScorrected_predicted_lesions_highlighted.png'

# Load and normalize the new image
print("Loading and normalizing new image...")
new_image = load_image(new_image_path)
new_image = np.expand_dims(new_image, axis=-1)  # Add channel dimension

# Extract patches from the new image
print("Extracting patches from new image...")
patch_size = (32, 32, 32)
stride = (16, 16, 16)
patches = extract_patches_from_image(new_image, patch_size=patch_size, stride=stride)
print(f"Patches extracted from new image. Number of patches: {len(patches)}")

# Make predictions on the new image patches
print("Making predictions on new image patches...")
predicted_patches = best_model.predict(np.array(patches))
predicted_patches = np.squeeze(predicted_patches)  # Remove channel dimension
predicted_patches = (predicted_patches > 0.5).astype(np.uint8)  # Convert to binary values

# Reconstruct the lesion mask from patches
print("Reconstructing lesion mask from patches...")
lesion_mask = np.zeros(new_image.shape[:3], dtype=np.uint8)

# Insert the predicted patches back into the lesion mask
patch_idx = 0
for i in range(0, new_image.shape[0] - patch_size[0] + 1, stride[0]):
    for j in range(0, new_image.shape[1] - patch_size[1] + 1, stride[1]):
        for k in range(0, new_image.shape[2] - patch_size[2] + 1, stride[2]):
            lesion_mask[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]] = np.maximum(
                lesion_mask[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]],
                predicted_patches[patch_idx]
            )
            patch_idx += 1

# Save the predicted lesion mask
print(f"Saving predicted lesion mask to {output_path}...")
nib.save(nib.Nifti1Image(lesion_mask, np.eye(4)), output_path)

# Create and save a highlighted image with lesions overlaid
print(f"Creating and saving highlighted image to {highlighted_output_path}...")
highlighted_image = new_image[..., 0].copy()  # Remove channel dimension
highlighted_image[lesion_mask > 0] = np.max(highlighted_image)  # Highlight lesions

plt.imshow(np.max(highlighted_image, axis=2), cmap='gray')  # Display maximum projection
plt.axis('off')
plt.savefig(highlighted_output_path, bbox_inches='tight', pad_inches=0)
plt.close()

print("Lesion detection and highlighting complete.")
