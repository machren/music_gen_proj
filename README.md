# AI Music Generation with GANs and VAEs

This project uses Deep Learning models to generate polyphonic piano music. It implements a Convolutional Neural Network (CNN) based Generative Adversarial Network (GAN) and a Variational Autoencoder (VAE) to learn from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) and create new musical pieces.

## Features

- **Two Advanced Models**: Implements both a GAN and a VAE for music generation.
- **Data Preprocessing**: Includes scripts to convert MIDI files into a piano roll format suitable for training.
- **Modular Structure**: The code is organized into logical files for easy configuration, training, and evaluation.
- **Evaluation**: Uses the Fréchet Inception Distance (FID) to measure the quality of generated music.

***

## File Structure

The project is organized into several Python files, each with a specific purpose:

-   `main.py`: The main script to run the entire pipeline, from data preprocessing to training and evaluation.
-   `config.py`: A centralized file for all configurations, such as file paths and model hyperparameters.
-   `data_preprocessing.py`: Contains all functions for loading MIDI files, converting them to piano roll matrices, and creating PyTorch DataLoaders.
-   `models.py`: Defines the neural network architectures for the Generator, Discriminator, and VAE.
-   `train.py`: Contains the training loops for the GAN and VAE models.
-   `evaluation.py`: Implements the FID score for evaluating the quality of generated samples.
-   `visualization.py`: Helper functions for plotting training losses and visualizing piano rolls.
-   `requirements.txt`: A list of all the Python libraries needed to run the project.

***

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone music_gen_proj
cd <repository-name>
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

Your project structure should look like this:
```arduino
your-project/
├── maestro-v3.0.0/
│   └── ... (dataset files)
├── config.py
├── main.py
└── ... (other project files)
```
*** 

## Usage

### 1. Configure Paths

Before running, open the config.py file and verify that all the paths match your directory structure.

```python

# config.py

--- Directories ---

GIVEN_DATA_DIR = 'maestro-v3.0.0'

DATASET_DIR = 'melody_matrices'

METADATA_FILE = 'maestro-v3.0.0/maestro-v3.0.0.csv'

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
