import csv
import urllib.request
import zipfile

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

DATA_URL = (
    "https://storage.googleapis.com/"
    "tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

class SequenceDataset(Dataset):
    def __init__(
        self,
        series,
        sequence_length=24,
    ):
        self.series = torch.tensor(
            series,
            dtype=torch.float32,
        )

        self.sequence_length = sequence_length

    def __len__(self):
        return (
            len(self.series)
            - self.sequence_length
        )

    def __getitem__(self, index):
        x = self.series[
            index:
            index + self.sequence_length
        ]

        y = self.series[
            index + self.sequence_length
        ]

        # [sequence_length] -> [sequence_length, 1]
        x = x.unsqueeze(-1)

        # escalar -> [1]
        y = y.unsqueeze(0)

        return x, y

def download_jena(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        data_dir
        / "jena_climate_2009_2016.csv"
    )

    if csv_path.exists():
        return csv_path

    zip_path = (
        data_dir
        / "jena_climate_2009_2016.csv.zip"
    )

    print("Downloading Jena Climate...")

    urllib.request.urlretrieve(
        DATA_URL,
        zip_path,
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_file:
        zip_file.extractall(data_dir)

    return csv_path

def load_temperature(csv_path):
    values = []

    with open(
        csv_path,
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            values.append(
                float(row["T (degC)"])
            )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    # El dataset original está muestreado
    # aproximadamente cada 10 minutos.
    # Tomamos una muestra por hora.
    values = values[::6]

    return values

def get_jena_loaders(
    data_dir,
    batch_size=64,
    sequence_length=24,
):
    csv_path = download_jena(
        data_dir
    )

    values = load_temperature(
        csv_path
    )

    dataset_size = len(values)

    train_end = int(
        dataset_size * 0.70
    )

    val_end = int(
        dataset_size * 0.85
    )

    train_values = values[:train_end]

    val_values = values[
        train_end:val_end
    ]

    test_values = values[val_end:]

    # Estadísticas calculadas solamente
    # con el conjunto de entrenamiento.
    mean = train_values.mean()
    std = train_values.std()

    train_values = (
        train_values - mean
    ) / std

    val_values = (
        val_values - mean
    ) / std

    test_values = (
        test_values - mean
    ) / std

    train_dataset = SequenceDataset(
        train_values,
        sequence_length,
    )

    val_dataset = SequenceDataset(
        val_values,
        sequence_length,
    )

    test_dataset = SequenceDataset(
        test_values,
        sequence_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    )
