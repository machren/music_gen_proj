# config.py

# --- Directories ---
GIVEN_DATA_DIR = '/workspace/maestro-v3.0.0'
DATASET_DIR = 'workspace/melody_matrices'
METADATA_FILE = '/workspace/maestro-v3.0.0/maestro-v3.0.0.csv'
GENERATED_IMAGES_DIR = 'generated_images'
GENERATED_MUSIC_DIR = 'generated_music'


# --- Data Preprocessing ---
FS = 100
PITCH_RANGE = [24, 96]
CHUNK_SIZE = 128
MAKE_DATASET_DIR = True


# --- Model Parameters ---
# GAN
Z_DIM = 100
N_FEATURES = 128
N_CHANNELS = 1

# --- VAE ---
N_FEATURES_VAE = 64
Z_DIM_VAE = 100


# --- Training Parameters ---
BATCH_SIZE = 64
N_EPOCHS = 10
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
