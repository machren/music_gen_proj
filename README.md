# AI Music Generation with GANs, VAEs, and VAE-GANs

This project uses deep learning models to generate polyphonic piano music. It implements three advanced generative architectures:

- **GAN** (Generative Adversarial Network)
- **VAE** (Variational Autoencoder)
- **VAE-GAN** (a hybrid model combining VAE and GAN advantages)

All models are trained on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) to learn musical structure and generate new, creative piano pieces.

***

## Features

- **Three Powerful Models**: Train and compare GAN, VAE, and VAE-GAN architectures for music generation.
- **Comprehensive Data Preprocessing**: MIDI files are converted into piano roll matrices, filtered by pitch range, and chunked for efficient training.
- **Modular Codebase**: Organized into distinct Python files for easy maintenance, extension, and experimentation.
- **Evaluation with Fréchet Inception Distance (FID)**: Quantitative metric to assess the quality and diversity of generated samples.
- **Robust Dataset Handling**: Handles corrupted or empty data gracefully with custom PyTorch datasets and collate functions.
- **Flexible Training Pipeline**: Configure and run training for any of the three models via a single main script.

***

## File Structure

The project is organized into several Python files, each with a specific purpose:

```arduino
your-project/
├── maestro-v3.0.0/ # Raw MAESTRO dataset files
├── melody_matrices/ # Preprocessed piano roll chunks saved as .npy
├── generated_images/ # Generated piano roll images by model
│ ├── gan/
│ ├── vae/
│ └── vae-gan/
├── generated_music/ # Generated MIDI files by model
│ ├── gan/
│ ├── vae/
│ └── vae-gan/
├── config.py # Project configuration and hyperparameters
├── main.py # Main script for preprocessing, training, evaluation
├── data_preprocessing.py # MIDI to piano roll conversion and Dataset classes
├── models.py # GAN, VAE, and VAE-GAN neural network architectures
├── train.py # Training loops for all models
├── evaluation.py # FID and other evaluation metrics
├── visualization.py # Plotting and piano roll visualization utilities
├── requirements.txt # Python dependencies
└── README.md # This file
```

***

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/machren/music_gen_proj
cd music_gen_proj
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required libraries using the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the MAESTRO v3.0.0 dataset from the official website. Unzip the folder and place it in the project's root directory.

*** 

## Usage

### 1. Configure Paths

Before running, open the config.py file and verify that all the paths match your directory structure, check training parameters, and choose which model to train.

```python

# config.py

--- Directories ---

GIVEN_DATA_DIR = 'maestro-v3.0.0'

DATASET_DIR = 'melody_matrices'

METADATA_FILE = 'maestro-v3.0.0/maestro-v3.0.0.csv'

# Model selection: 'gan', 'vae', or 'vae-gan'
MODEL_TYPE = 'vae-gan'

```

### 2. Run the Main Script

Execute the main.py script to start the data preprocessing and model training.

```bash
python3 main.py
```

The script will first process all MIDI files and save them as NumPy arrays in the melody_matrices directory. After that, it will begin training the GAN and VAE models.


### 3. View the Output

Generated Images: Images of generated piano rolls will be saved in the generated_images/ directory.

Generated Music: Generated MIDI files will be saved in the generated_music/ directory.

## Notes

The VAE-GAN model combines the encoding and sampling abilities of the VAE with the adversarial training of GANs to improve generation quality and diversity.
- You can switch models by simply changing 'MODEL_TYPE' in config.py without modifying the code.
- The codebase is modular, allowing you to add new models, training strategies, or evaluation metrics easily.
- Use the visualization.py utilities to inspect piano roll outputs and training curves.
