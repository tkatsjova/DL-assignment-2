from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.data_loading import (
    load_folder_metadata,
    load_h5_file,
    extract_label_from_filename,
    extract_subject_id_from_filename,
    ID_TO_LABEL,
)
from src.data.preprocessing import preprocess_recording


class MEGWindowDataset(Dataset):
    def __init__(
        self,
        original_sampling_rate: int = 2034,
        downsample_factor: int = 4,
        window_seconds: float = 2.0,
        overlap: float = 0.5,
        folder: Path | None = None,
        files: list[Path] | None = None,
        max_files: int | None = None,
    ) -> None:
        # Accept either a folder (scan all files inside)
        # or an explicit list of files (e.g. a chunk for cross-subject training).
        # This lets the same class handle both scenarios without duplicating code.
        if files is not None:
            metadata = [
                {
                    "file_path": f,
                    "label": extract_label_from_filename(f),
                    "label_name": ID_TO_LABEL[extract_label_from_filename(f)],
                    "subject_id": extract_subject_id_from_filename(f),
                }
                for f in files
            ]
        elif folder is not None:
            metadata = load_folder_metadata(folder)
        else:
            raise ValueError("Provide either 'folder' or 'files'")

        if max_files is not None:
            metadata = metadata[:max_files]

        self.folder = folder
        self.metadata = metadata
        self.file_names = []
        self.subject_ids = []

        windows_per_file = []
        labels_per_file = []

        source = folder if folder is not None else f"{len(metadata)} files"
        print(f"\nBuilding dataset from: {source}")
        print(f"Number of files: {len(metadata)}")

        for item in tqdm(metadata, desc="Preprocessing files"):
            x = load_h5_file(item["file_path"])

            windows, labels = preprocess_recording(
                x=x,
                label=item["label"],
                original_sampling_rate=original_sampling_rate,
                downsample_factor=downsample_factor,
                window_seconds=window_seconds,
                overlap=overlap,
            )

            windows_per_file.append(windows)
            labels_per_file.append(labels)

            self.file_names.extend([item["file_path"].name] * len(labels))
            self.subject_ids.extend([item["subject_id"]] * len(labels))

        self.X = np.concatenate(windows_per_file, axis=0).astype(np.float32)
        self.y = np.concatenate(labels_per_file, axis=0).astype(np.int64)

        print(f"Final X shape: {self.X.shape}")
        print(f"Final y shape: {self.y.shape}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def create_dataloader(
    dataset: MEGWindowDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
