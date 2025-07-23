# config.py

# --- Directories ---
GIVEN_DATA_DIR = '/workspace/maestro-v3.0.0'
DATASET_DIR = '/workspace/melody_matrices'
METADATA_FILE = '/workspace/maestro-v3.0.0/maestro-v3.0.0.csv'
GENERATED_ROLLS_DIR = "./generated_rolls"
GENERATED_MIDI_DIR = "./generated_midis"
GENERATED_IMAGES_DIR = "./generated_images"
LOG_DIR = '/workspace/logs'
MOD_DIR = '/workspace/models'

# --- Data Preprocessing ---
FS = 25
PITCH_RANGE = [24, 96]
CHUNK_SIZE = 256
MAKE_DATASET_DIR = False # IMPORTANT Set to True if you want to prepate Numpy dataset
FRACTION = 0.1 # IMPORTANT Change the number to take a subset of dataset [0.0,1.0]

# --- Model Parameters ---

TRAIN_MODE = "vae_gan" # IMPORTANT Set to "gan" // "vae" // "vae_gan" depending on the task

# GAN
Z_DIM = 100
N_FEATURES = 128
N_CHANNELS = 1

# VAE 
N_FEATURES_VAE = 64
Z_DIM_VAE = 100

# --- Training Parameters ---
BATCH_SIZE = 64
N_EPOCHS = 1
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# --- Generation Parameters ---
N_SAMPLES = 10
