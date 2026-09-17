from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("boolode_data/raw/dyn-LL-1/simulations")


# ============================================================
# CSVファイルの取得
# ============================================================

csv_files = sorted(
    DATA_DIR.glob("E*.csv"),
    key=lambda path: int(path.stem[1:])
)

print("===== ファイル確認 =====")
print(f"ファイル数: {len(csv_files)}")

if len(csv_files) != 300:
    raise ValueError(
        f"シミュレーション数が300ではありません: {len(csv_files)}"
    )


# ============================================================
# 基準情報
# ============================================================

first_df = pd.read_csv(csv_files[0], index_col=0)

gene_names = list(first_df.index)
reference_time_names = list(first_df.columns)

num_genes = len(gene_names)
num_timesteps = len(reference_time_names)

# E0_1 → 1、E0_2 → 2 のように時間番号だけを抽出
reference_time_numbers = [
    int(name.split("_")[-1])
    for name in reference_time_names
]

print("")
print("===== 基準ファイル =====")
print(f"ファイル: {csv_files[0]}")
print(f"遺伝子数: {num_genes}")
print(f"時系列長: {num_timesteps}")
print(f"遺伝子名: {gene_names}")
print(f"時間番号: {reference_time_numbers[:5]} ...")
print(f"最後の時間番号: {reference_time_numbers[-5:]}")


# ============================================================
# 全CSVの読み込み・構造確認
# ============================================================

all_data = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file, index_col=0)

    # 遺伝子名の確認
    if list(df.index) != gene_names:
        raise ValueError(
            f"遺伝子の並びが異なります: {csv_file}"
        )

    # 時間番号の確認
    current_time_numbers = [
        int(name.split("_")[-1])
        for name in df.columns
    ]

    if current_time_numbers != reference_time_numbers:
        raise ValueError(
            f"時間番号の並びが異なります: {csv_file}"
        )

    # データ形状の確認
    if df.shape != (num_genes, num_timesteps):
        raise ValueError(
            f"形状が異なります: {csv_file}, shape={df.shape}"
        )

    all_data.append(
        df.to_numpy(dtype=np.float32)
    )


# ============================================================
# 配列化
# ============================================================

X = np.stack(all_data, axis=0)

print("")
print("===== 読み込み結果 =====")
print(f"X.shape: {X.shape}")
print(f"X.dtype: {X.dtype}")
print(f"最小値: {X.min()}")
print(f"最大値: {X.max()}")
print(f"平均値: {X.mean()}")
print(f"標準偏差: {X.std()}")


# ============================================================
# 欠損値・無限値の確認
# ============================================================

print("")
print("===== 欠損値・無限値の確認 =====")
print(f"NaNの数: {np.isnan(X).sum()}")
print(f"無限値の数: {np.isinf(X).sum()}")


# ============================================================
# 先頭データの確認
# ============================================================

print("")
print("===== E0.csvの先頭データ =====")
print(X[0, :, :5])


# ============================================================
# NRI入力形式
# ============================================================

X_nri = X[..., np.newaxis]

print("")
print("===== NRI入力形式 =====")
print(f"発現量のみ: {X_nri.shape}")
print("形式: [実験数, 遺伝子数, 時系列長, 特徴量数]")
