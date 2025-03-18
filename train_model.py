import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from data_preparation import load_all_images
from combined_script import extract_patches_from_all_images, augment_patches
from cnn_model_3d import CNN3DHyperModel
from kerastuner.tuners import RandomSearch

# Set XLA_FLAGS environment variable with the correct CUDA path
cuda_path = "/opt/nesi/CS400_centos7_bdw/CUDA/12.3.0"
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_path}'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f"XLA_FLAGS set to specify the CUDA data directory: {cuda_path}")
print("TF_GPU_ALLOCATOR set to cuda_malloc_async")

# Enable memory growth for the GPU
print("Setting up GPU configuration...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Disable mixed precision training
# policy = Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)
print("Mixed precision policy disabled.")

# Disable XLA compilation
tf.config.optimizer.set_jit(False)
print("XLA compilation disabled.")

# Define file paths
flair_paths = [f'/nesi/project/uoa04272/software/tensorflow-2.17.0/BPS/BPS_OUTPUT_IMAGES/Patient_{i}_rr_mni_flair_bet_bias_corrected.nii.gz' for i in range(1, 71)]
mask_paths = [f'/nesi/project/uoa04272/software/tensorflow-2.17.0/BPS/BPS_OUTPUT_IMAGES/Patient_{i}_rr_mni_lesion_bias_corrected.nii.gz' for i in range(1, 71)]

# Load and normalize all images
print("Loading and normalizing images...")
flair_images, mask_images = load_all_images(flair_paths, mask_paths)
print("Images loaded and normalized.")

# Extract patches with larger patch size and more stride to increase the number of patches
print("Extracting patches...")
patch_size = (32, 32, 32)
stride = (16, 16, 16)  # Increase stride to reduce the number of patches
patches, mask_patches = extract_patches_from_all_images(flair_images, mask_images, patch_size=patch_size, stride=stride)
print(f"Patches extracted. Number of patches: {len(patches)}, Number of mask patches: {len(mask_patches)}")

# Data augmentation
print("Applying data augmentation...")
augmented_patches, augmented_mask_patches = augment_patches(patches, mask_patches, max_augmentations=5000)  # Further reduce augmentations
print("Data augmentation applied.")

# Prepare data for training
print("Preparing data for training...")
try:
    X = np.expand_dims(augmented_patches, axis=-1)  # Add channel dimension
    y = np.expand_dims(augmented_mask_patches, axis=-1)  # Add channel dimension
    y = np.squeeze(y)  # Ensure y has shape (None, 32, 32, 32)
    y = np.mean(y, axis=(1, 2, 3))  # Convert y to shape (None, 1)
    y = y.astype(np.float32)  # Ensure y is of dtype float32
    print(f"Data prepared for training. X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"Error during data preparation: {e}")

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
try:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training and validation sets. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
except Exception as e:
    print(f"Error during data splitting: {e}")

# Further reduce batch size to reduce memory usage
batch_size = 2

# Create tf.data.Dataset for efficient data loading
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Setup model checkpoint, early stopping, and learning rate reduction
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
csv_logger = CSVLogger('training_log.csv')
print("Model checkpoint, early stopping, and learning rate reduction setup.")

# Clear TensorFlow session to free up memory between trials
def clear_tf_session():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    print("TensorFlow session cleared.")

# Hyperparameter tuning
hypermodel = CNN3DHyperModel()
tuner = RandomSearch(hypermodel, objective='val_accuracy', max_trials=10, executions_per_trial=2, directory='hyperparam_tuning', project_name='3d_cnn')

print("Starting hyperparameter search...")
for trial in range(10):
    clear_tf_session()
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(f'best_model_tuned_trial_{trial}.keras')
    print(f"Hyperparameter search complete for trial {trial} and best model saved.")

if __name__ == "__main__":
    print("Script execution complete.")
