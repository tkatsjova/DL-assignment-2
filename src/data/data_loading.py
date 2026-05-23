from pathlib import Path

import h5py
import numpy as np


LABEL_MAP = {
    "rest": 0,
    "task_story_math": 1,
    "task_working_memory": 2,
    "task_motor": 3,
}

ID_TO_LABEL = {
    0: "rest",
    1: "math",
    2: "memory",
    3: "motor",
}


def load_h5_file(file_path: Path) -> np.ndarray:
    name = file_path.stem

    with h5py.File(file_path, "r") as f:
        if name in f:
            data = f[name][()]
        else:
            # fallback in case the internal dataset name differs from the filename
            first_key = list(f.keys())[0]
            data = f[first_key][()]

    return np.asarray(data, dtype=np.float32)


def extract_label_from_filename(file_path: Path) -> int:
    name = file_path.stem

    if name.startswith("task_story_math"):
        return LABEL_MAP["task_story_math"]
    if name.startswith("task_working_memory"):
        return LABEL_MAP["task_working_memory"]
    if name.startswith("task_motor"):
        return LABEL_MAP["task_motor"]
    if name.startswith("rest"):
        return LABEL_MAP["rest"]

    raise ValueError(f"Unknown task label in file name: {file_path.name}")


def extract_subject_id_from_filename(file_path: Path) -> str:
    parts = file_path.stem.split("_")
    return parts[-2]


def list_h5_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.h5"))


def load_folder_metadata(folder: Path) -> list[dict]:
    metadata = []

    for file_path in list_h5_files(folder):
        label = extract_label_from_filename(file_path)

        metadata.append(
            {
                "file_path": file_path,
                "label": label,
                "label_name": ID_TO_LABEL[label],
                "subject_id": extract_subject_id_from_filename(file_path),
            }
        )

    return metadata


def inspect_folder(folder: Path, max_files: int = 5) -> None:
    metadata = load_folder_metadata(folder)

    print("\n" + "=" * 80)
    print(f"Folder: {folder}")
    print(f"Number of .h5 files: {len(metadata)}")

    if not metadata:
        print("No .h5 files found.")
        return

    labels = {}
    subjects = set()

    for item in metadata:
        labels[item["label_name"]] = labels.get(item["label_name"], 0) + 1
        subjects.add(item["subject_id"])

    print(f"Subjects: {sorted(subjects)}")
    print(f"Label counts: {labels}")

    print("\nExample files:")
    for item in metadata[:max_files]:
        x = load_h5_file(item["file_path"])

        print(
            f"- {item['file_path'].name} | "
            f"label={item['label_name']} | "
            f"subject={item['subject_id']} | "
            f"shape={x.shape}"
        )
