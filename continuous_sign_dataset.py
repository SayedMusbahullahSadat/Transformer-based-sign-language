import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class ContinuousSignDataset(Dataset):
    """
    On-the-fly dataset that concatenates a random number (1..N) of isolated-sign clips into one continuous sequence,
    with variable blank-frame gaps, Gaussian noise augmentation, and optional left-right mirroring.

    Args:
        csv_path: Path to the filled landmarks CSV (with columns video_id, label, and feature columns).
        N: Maximum number of signs per continuous example (will randomly sample 1..N).
        blank_token: String token representing silent/blank frames.
        blank_range: Tuple (min_blank, max_blank) for random number of blank frames between signs.
        noise_std: Standard deviation of Gaussian noise to add to landmarks.
        mirror_prob: Probability [0-1] of horizontally mirroring each clip.
        dataset_size: Total number of examples; if None, defaults to one example per video.
    """
    def __init__(self,
                 csv_path: str,
                 N: int = 3,
                 blank_token: str = '<blank>',
                 blank_range: tuple = (1, 3),
                 noise_std: float = 0.01,
                 mirror_prob: float = 0.5,
                 dataset_size: int = None):
        # Load the CSV
        df = pd.read_csv(csv_path)
        # Build per-video frame & label maps
        self.video_to_frames = {}
        self.video_to_label  = {}
        for vid, sub in df.groupby('video_id'):
            feats = sub.drop(columns=['video_id','label']).values.astype(np.float32)
            self.video_to_frames[vid] = feats
            self.video_to_label[vid]  = sub['label'].iat[0]
        self.ids = list(self.video_to_frames.keys())

        # Prepare label encoder including blank
        glosses = sorted(set(self.video_to_label.values()))
        if blank_token not in glosses:
            glosses.append(blank_token)
        self.encoder   = LabelEncoder().fit(glosses)
        self.blank_idx = int(self.encoder.transform([blank_token])[0])

        # Store settings
        self.N            = N
        self.blank_range  = blank_range
        self.noise_std    = noise_std
        self.mirror_prob  = mirror_prob
        self.dataset_size = dataset_size or len(self.ids)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Randomly choose number of signs between 1 and N
        num_signs = random.randint(1, self.N)
        sequences = []
        targets   = []

        for i in range(num_signs):
            vid = random.choice(self.ids)
            seq = self.video_to_frames[vid].copy()  # [T_i, feat_dim]

            # Add Gaussian noise
            if self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)
                seq += noise

            # Mirror augmentation (flip x-coordinates)
            if random.random() < self.mirror_prob:
                # x-axis is every 3rd element starting at index 0
                seq[:, 0::3] = 1.0 - seq[:, 0::3]

            sequences.append(seq)
            targets.append(int(self.encoder.transform([self.video_to_label[vid]])[0]))

            # Insert random-length blank gap if not last sign
            if i < num_signs - 1:
                gap = random.randint(self.blank_range[0], self.blank_range[1])
                blank_frame = np.zeros((gap, seq.shape[1]), dtype=np.float32)
                sequences.append(blank_frame)
                targets.append(self.blank_idx)

        # Stack all into full sequence
        full_seq = np.vstack(sequences)
        frames_tensor = torch.from_numpy(full_seq)           # shape: [T, feat_dim]
        target_tensor = torch.tensor(targets, dtype=torch.long)  # length = 2*num_signs-1
        return frames_tensor, target_tensor

# Collate function for CTC loss

def ctc_collate_fn(batch):
    feats = [item[0] for item in batch]
    tars  = [item[1] for item in batch]
    input_lengths  = [f.size(0) for f in feats]
    target_lengths = [t.size(0) for t in tars]
    max_T = max(input_lengths)
    feat_dim = feats[0].size(1)

    # Pad each feature sequence to max_T
    padded = []
    for f in feats:
        pad = torch.zeros((max_T - f.size(0), feat_dim), dtype=f.dtype)
        padded.append(torch.cat([f, pad], dim=0))
    stacked = torch.stack(padded, dim=0)  # [B, max_T, feat_dim]
    padded_inputs = stacked.permute(1, 0, 2)  # [max_T, B, feat_dim]

    # Flatten targets for CTC
    targets = torch.cat(tars, dim=0)
    return padded_inputs, targets, input_lengths, target_lengths
