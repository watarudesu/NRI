from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_boolode_data(
    data_dir="boolode_data/processed/dyn-LL-1",
    batch_size=8,
):
    data_dir = Path(data_dir)

    train_data = np.load(data_dir / "train.npy")
    valid_data = np.load(data_dir / "valid.npy")
    test_data = np.load(data_dir / "test.npy")

    print("===== NumPyデータ =====")
    print(f"train: {train_data.shape}")
    print(f"valid: {valid_data.shape}")
    print(f"test:  {test_data.shape}")

    train_tensor = torch.from_numpy(train_data).float()
    valid_tensor = torch.from_numpy(valid_data).float()
    test_tensor = torch.from_numpy(test_data).float()

    train_dataset = TensorDataset(train_tensor)
    valid_dataset = TensorDataset(valid_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, valid_loader, test_loader = load_boolode_data()

    batch = next(iter(train_loader))
    data = batch[0]

    print("")
    print("===== DataLoader確認 =====")
    print(f"バッチ形状: {data.shape}")
    print(f"データ型: {data.dtype}")
