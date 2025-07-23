# data_preprocessing.py

import os
import glob
import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch.utils.data import Subset
import random

import config

def get_piano_roll(pm, pedal_threshold=64):
    '''
    Returns a piano roll matrix from a pretty_midi object.
    '''
    try:
        instruments = [i for i in pm.instruments if not i.is_drum]
        if not instruments:
            return None, None

        instruments.sort(key=lambda x: -max([note.pitch for note in x.notes if note.pitch is not None] or [0]))
        
        melody_instrument = instruments[0]

        for note in melody_instrument.notes:
            note.velocity = 100 if note.velocity > 0 else 0

        try:
            fs = int(1 / pm.get_tick_length())
        except (AttributeError, ZeroDivisionError):
            fs = config.FS 

        piano_roll = melody_instrument.get_piano_roll(fs=fs, pedal_threshold=pedal_threshold)
        return piano_roll, fs
    except Exception as e:
        print(f"    - Error processing piano roll: {e}")
        return None, None


def prepare_dataset(filenames, metadata, dataset_dir, make_dir=True, fs=25, pitch_range=[24, 96], chunk_size=128):
    '''
    Creates a dataset of piano roll chunks and saves them as numpy arrays.
    '''
    print("PREPARING DATASET....")
    if make_dir:
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'validation'), exist_ok=True)

    for f in filenames:
        try:
            base_name = os.path.basename(f)
            split_series = metadata[metadata.midi_filename.str.contains(base_name.replace('.midi', ''))]['split']
            if split_series.empty:
                # print(f"Skipping {base_name}: No metadata found.")
                continue
            split = split_series.values[0]

            pm = pretty_midi.PrettyMIDI(f)
            piano_roll, _ = get_piano_roll(pm)

            if piano_roll is None:
                # print(f"Skipping {base_name} due to processing error.")
                continue

            if piano_roll.shape[0] < pitch_range[1]:
                 # print(f"Skipping {base_name}: piano roll pitch range is too small ({piano_roll.shape[0]})")
                 continue
            
            piano_roll = piano_roll[pitch_range[0]:pitch_range[1], :]
            
            if piano_roll.shape[1] < chunk_size:
                # print(f"Skipping {base_name}: piano roll is not long enough to create a chunk.")
                continue

            n_chunks = piano_roll.shape[1] // chunk_size

            for i in range(n_chunks):
                chunk = piano_roll[:, i * chunk_size:(i + 1) * chunk_size].astype(np.float32)
                save_path = os.path.join(dataset_dir, split, f"{os.path.splitext(base_name)[0]}_{i}.npy")
                np.save(save_path, chunk)

        except (OSError, IOError) as e:
            print(f"Stopping data preparation due to file system error: {e}")
            return
        except Exception as e:
            print(f"Failed to process {f}: {e}")

class PianoRollDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = glob.glob(os.path.join(root_dir, '*.npy'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        try:
            piano_roll = np.load(self.filenames[idx])
            # Check if the file is empty
            if piano_roll.size == 0:
                return None
            if self.transform:
                piano_roll = self.transform(piano_roll)
            return piano_roll
        except (EOFError, IOError):
            # Return None if the file is corrupted
            return None

# This function will filter out the None values from the dataset
def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)


def get_dataloaders(fraction=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load full datasets
    train_dataset = PianoRollDataset(os.path.join(config.DATASET_DIR, 'train'), transform=transform)
    test_dataset = PianoRollDataset(os.path.join(config.DATASET_DIR, 'test'), transform=transform)
    validation_dataset = PianoRollDataset(os.path.join(config.DATASET_DIR, 'validation'), transform=transform)

    # If fraction < 1.0, take a subset of each
    def get_subset(dataset):
        if fraction >= 1.0:
            return dataset
        n = len(dataset)
        subset_size = int(fraction * n)
        indices = random.sample(range(n), subset_size)
        return Subset(dataset, indices)

    train_dataset = get_subset(train_dataset)
    test_dataset = get_subset(test_dataset)
    validation_dataset = get_subset(validation_dataset)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_filter_none)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_filter_none)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_filter_none)

    return train_loader, test_loader, validation_loader
