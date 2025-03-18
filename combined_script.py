import numpy as np
import random
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
from data_preparation import load_all_images
import os
import matplotlib.pyplot as plt

def extract_patches(image, mask=None, patch_size=(32, 32, 32), stride=(16, 16, 16)):
    patches = []
    mask_patches = []
    img_shape = image.shape

    print(f"Starting patch extraction: image shape {img_shape}, patch size {patch_size}, stride {stride}")

    for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                patches.append(patch)
                if mask is not None:
                    mask_patch = mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    mask_patches.append(mask_patch)
    
    print(f"Finished patch extraction: extracted {len(patches)} patches")
    if mask is not None:
        return np.array(patches), np.array(mask_patches)
    return np.array(patches)

def extract_patches_from_all_images(flair_images, mask_images=None, patch_size=(32, 32, 32), stride=(16, 16, 16)):
    all_patches = []
    all_mask_patches = []
    for i, flair_image in enumerate(flair_images):
        print(f"Extracting patches from image {i + 1}/{len(flair_images)}")
        if mask_images is not None:
            patches, mask_patches = extract_patches(flair_image, mask_images[i], patch_size, stride)
            all_mask_patches.append(mask_patches)
        else:
            patches = extract_patches(flair_image, patch_size=patch_size, stride=stride)
        all_patches.append(patches)
    
    print("Finished extracting patches from all images")
    if mask_images is not None:
        return np.concatenate(all_patches), np.concatenate(all_mask_patches)
    return np.concatenate(all_patches)

def augment_image(image, mask, target_shape=(32, 32, 32)):
    angle = random.uniform(-10, 10)
    image = rotate(image, angle, axes=(0, 1), reshape=False)
    mask = rotate(mask, angle, axes=(0, 1), reshape=False)
    
    zoom_factor = random.uniform(0.9, 1.1)
    image = zoom(image, zoom_factor)
    mask = zoom(mask, zoom_factor)
    
    if random.random() > 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    
    if image.shape != target_shape:
        image = resize(image, target_shape, preserve_range=True, anti_aliasing=True)
    if mask.shape != target_shape:
        mask = resize(mask, target_shape, preserve_range=True, anti_aliasing=True)

    return image, mask

def augment_patches(patches, mask_patches, target_shape=(32, 32, 32), max_augmentations=10000):
    augmented_patches = []
    augmented_mask_patches = []
    for i, (patch, mask_patch) in enumerate(zip(patches, mask_patches)):
        if i >= max_augmentations:
            break
        if i % 100 == 0:
            print(f"Augmenting patch {i}/{min(len(patches), max_augmentations)}")
        aug_patch, aug_mask_patch = augment_image(patch, mask_patch, target_shape)
        augmented_patches.append(aug_patch)
        augmented_mask_patches.append(aug_mask_patch)
    
    print("Finished augmenting patches")
    return np.array(augmented_patches), np.array(augmented_mask_patches)

def save_random_patches_and_augmentations(patches, mask_patches, augmented_patches, augmented_mask_patches, output_dir, num_samples=25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate indices based on the length of augmented_patches
    indices = random.sample(range(len(augmented_patches)), num_samples)
    
    for i, idx in enumerate(indices):
        fig, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(patches[idx][16, :, :], cmap='gray')
        axarr[0, 0].set_title('Original Patch')
        axarr[0, 1].imshow(mask_patches[idx][16, :, :], cmap='gray')
        axarr[0, 1].set_title('Original Mask')
        axarr[1, 0].imshow(augmented_patches[idx][16, :, :], cmap='gray')
        axarr[1, 0].set_title('Augmented Patch')
        axarr[1, 1].imshow(augmented_mask_patches[idx][16, :, :], cmap='gray')
        axarr[1, 1].set_title('Augmented Mask')
        
        plt.savefig(os.path.join(output_dir, f'sample_{i}.png'))
        plt.close()

def extract_patches_from_image(image, patch_size=(32, 32, 32), stride=(16, 16, 16)):
    patches = []
    img_shape = image.shape

    print(f"Starting patch extraction: image shape {img_shape}, patch size {patch_size}, stride {stride}")

    for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
            for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
                patch = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                patches.append(patch)
    
    print(f"Finished patch extraction: extracted {len(patches)} patches")
    return np.array(patches)

def reconstruct_image_from_patches(patches, image_shape, patch_size, stride):
    reconstructed_image = np.zeros(image_shape)
    count_image = np.zeros(image_shape)
    
    patch_index = 0
    for i in range(0, image_shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, image_shape[1] - patch_size[1] + 1, stride[1]):
            for k in range(0, image_shape[2] - patch_size[2] + 1, stride[2]):
                reconstructed_image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += patches[patch_index]
                count_image[i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]] += 1
                patch_index += 1

    count_image[count_image == 0] = 1  # Avoid division by zero
    reconstructed_image /= count_image
    return reconstructed_image

# Example usage
if __name__ == "__main__":
    flair_paths = [f'/nesi/project/uoa04272/software/tensorflow-2.17.0/BPS/BPS_OUTPUT_IMAGES/Patient_{i}_rr_mni_flair_bet_bias_corrected.nii.gz' for i in range(1, 71)]
    mask_paths = [f'/nesi/project/uoa04272/software/tensorflow-2.17.0/BPS/BPS_OUTPUT_IMAGES/Patient_{i}_rr_mni_lesion_bias_corrected.nii.gz' for i in range(1, 71)]
    
    flair_images, mask_images = load_all_images(flair_paths, mask_paths)
    patches, mask_patches = extract_patches_from_all_images(flair_images, mask_images)
    
    augmented_patches, augmented_mask_patches = augment_patches(patches, mask_patches, max_augmentations=5000)
    print("Applied augmentation to patches.")
    
    output_dir = '/nesi/project/uoa04272/software/tensorflow-2.17.0/3D_CNN_MS_Detection/patches_and_augmentations'
    save_random_patches_and_augmentations(patches, mask_patches, augmented_patches, augmented_mask_patches, output_dir)
    print(f'Saved random patches and augmentations to {output_dir}')
