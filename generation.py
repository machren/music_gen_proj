import pretty_midi
import numpy as np
import os
import torch

def piano_roll_to_midi(piano_roll, fs=4, program=0, threshold=0.5):
    """
    Convert a piano roll into a PrettyMIDI object.
    """
    if isinstance(piano_roll, torch.Tensor):
        piano_roll = piano_roll.detach().cpu().numpy()

    piano_roll = np.squeeze(piano_roll)

    # Binarize if needed
    if piano_roll.max() > 1.0:
        piano_roll = piano_roll / piano_roll.max()
    piano_roll = (piano_roll > threshold).astype(np.int32)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    n_pitches, n_frames = piano_roll.shape
    dt = 1.0 / fs 

    for pitch in range(n_pitches):
        note_on = None
        for t in range(n_frames):
            if piano_roll[pitch, t] == 1:
                if note_on is None:
                    note_on = t * dt
            else:
                if note_on is not None:
                    note_off = t * dt
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=note_on,
                        end=note_off
                    )
                    instrument.notes.append(note)
                    note_on = None

        if note_on is not None:
            note_off = n_frames * dt
            instrument.notes.append(
                pretty_midi.Note(velocity=100, pitch=pitch, start=note_on, end=note_off)
            )

    pm.instruments.append(instrument)
    return pm

def convert_saved_piano_rolls_to_midis(piano_roll_dir, midi_dir, fs=4):
    os.makedirs(midi_dir, exist_ok=True)
    for filename in os.listdir(piano_roll_dir):
        if filename.endswith(".npy"):
            roll_path = os.path.join(piano_roll_dir, filename)
            piano_roll = np.load(roll_path)
            midi = piano_roll_to_midi(piano_roll[0], fs=fs)
            midi_filename = filename.replace(".npy", ".mid")
            midi_path = os.path.join(midi_dir, midi_filename)
            midi.write(midi_path)

def save_generated_piano_rolls(vae, z_dim, n_samples, device, save_dir, model_type):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n_samples):
        
        if model_type in ("vae", "vae_gan"):
            z = torch.randn(1, z_dim).to(device)
            x_gen = vae.decode(z)
            
        elif model_type == "gan":
            z = torch.randn(1, z_dim, 1, 1).to(device)
            x_gen = vae(z)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        with torch.no_grad():
            piano_roll = x_gen.cpu().numpy()
        np.save(os.path.join(save_dir, f'piano_roll_{i}.npy'), piano_roll)
