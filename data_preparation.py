import nibabel as nib
import numpy as np

def load_image(filepath):
    """Load and normalize a NIfTI image from the given filepath."""
    image = nib.load(filepath).get_fdata()
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def load_all_images(flair_paths, mask_paths):
    """Load and normalize all FLAIR and mask images."""
    flair_images = [load_image(path) for path in flair_paths]
    mask_images = [load_image(path) for path in mask_paths]
    return flair_images, mask_images

def extract_patches_from_all_images(flair_images, mask_images, patch_size, stride):
    """Extract patches from all images."""
    patches = []
    mask_patches = []
    for flair_image, mask_image in zip(flair_images, mask_images):
        for i in range(0, flair_image.shape[0] - patch_size[0] + 1, stride[0]):
            for j in range(0, flair_image.shape[1] - patch_size[1] + 1, stride[1]):
                for k in range(0, flair_image.shape[2] - patch_size[2] + 1, stride[2]):
                    patch = flair_image[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                    mask_patch = mask_image[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2]]
                    patches.append(patch)
                    mask_patches.append(mask_patch)
    return patches, mask_patches

def augment_patches(patches, mask_patches, max_augmentations):
    """Data augmentation for patches."""
    augmented_patches = patches.copy()
    augmented_mask_patches = mask_patches.copy()

    for _ in range(max_augmentations):
        idx = np.random.randint(len(patches))
        patch = patches[idx]
        mask_patch = mask_patches[idx]

        # Example augmentation: flip the patch
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0)
            mask_patch = np.flip(mask_patch, axis=0)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1)
            mask_patch = np.flip(mask_patch, axis=1)
        if np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2)
            mask_patch = np.flip(mask_patch, axis=2)

        augmented_patches.append(patch)
        augmented_mask_patches.append(mask_patch)

    return augmented_patches, augmented_mask_patches
