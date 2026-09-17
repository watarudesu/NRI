from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 設定
# ============================================================

DATA_DIR = Path("boolode_data/raw/dyn-LL-1/simulations")
OUTPUT_DIR = Path("boolode_data/processed/dyn-LL-1")

SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CSVファイルの読み込み
# ============================================================

csv_files = sorted(
    DATA_DIR.glob("E*.csv"),
    key=lambda path: int(path.stem[1:])
)

all_data = []

gene_names = None
time_numbers = None

for csv_file in csv_files:
    df = pd.read_csv(csv_file, index_col=0)

    current_gene_names = list(df.index)
    current_time_numbers = [
        int(name.split("_")[-1])
        for name in df.columns
    ]

    if gene_names is None:
        gene_names = current_gene_names
        time_numbers = current_time_numbers
    else:
        if current_gene_names != gene_names:
            raise ValueError(
                f"遺伝子の並びが異なります: {csv_file}"
            )

        if current_time_numbers != time_numbers:
            raise ValueError(
                f"時間番号の並びが異なります: {csv_file}"
            )

    all_data.append(
        df.to_numpy(dtype=np.float32)
    )

X = np.stack(all_data, axis=0)

print("===== 読み込み結果 =====")
print(f"X.shape: {X.shape}")


# ============================================================
# NRI入力形式に変換
# ============================================================

X = X[..., np.newaxis]

print(f"NRI入力形式: {X.shape}")


# ============================================================
# データ分割
# ============================================================

num_samples = X.shape[0]

rng = np.random.default_rng(SEED)
indices = rng.permutation(num_samples)

num_train = int(num_samples * TRAIN_RATIO)
num_valid = int(num_samples * VALID_RATIO)

train_indices = indices[:num_train]
valid_indices = indices[num_train:num_train + num_valid]
test_indices = indices[num_train + num_valid:]

X_train = X[train_indices]
X_valid = X[valid_indices]
X_test = X[test_indices]


# ============================================================
# 保存
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "train.npy", X_train)
np.save(OUTPUT_DIR / "valid.npy", X_valid)
np.save(OUTPUT_DIR / "test.npy", X_test)

np.save(OUTPUT_DIR / "train_indices.npy", train_indices)
np.save(OUTPUT_DIR / "valid_indices.npy", valid_indices)
np.save(OUTPUT_DIR / "test_indices.npy", test_indices)

with open(OUTPUT_DIR / "gene_names.txt", "w") as f:
    for gene_name in gene_names:
        f.write(gene_name + "\n")

with open(OUTPUT_DIR / "README.txt", "w") as f:
    f.write("BoolODE processed data\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Train samples: {len(train_indices)}\n")
    f.write(f"Valid samples: {len(valid_indices)}\n")
    f.write(f"Test samples: {len(test_indices)}\n")
    f.write(f"Shape: {X.shape}\n")
    f.write("Format: [samples, genes, timesteps, features]\n")
    f.write("Features: expression level only\n")


# ============================================================
# 結果確認
# ============================================================

print("")
print("===== 分割結果 =====")
print(f"学習データ: {X_train.shape}")
print(f"検証データ: {X_valid.shape}")
print(f"テストデータ: {X_test.shape}")

print("")
print("===== 保存先 =====")
print(OUTPUT_DIR)

print("")
print("データ分割と保存が完了しました。")
