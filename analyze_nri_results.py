import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

# 出力先
OUTPUT_DIR = "analysis_results"

# ------------------------------------------------------------
# ① 粒子数比較
#    学習データ数 = 10000
# ------------------------------------------------------------

PARTICLE_COMPARISON = [
    ("5particle", 5, 10000),
    ("10particle", 10, 10000),
    ("15particle", 15, 10000),
    ("20particle",20,10000),
]

# ------------------------------------------------------------
# ② 5粒子・学習データ数比較
#    100未満は除外
# ------------------------------------------------------------

DATA_SIZE_COMPARISON = [
    100,
    250,
    500,
    1000,
    2000,
    3000,
    10000,
]


# ============================================================
# log.txt 読み込み
# ============================================================

def read_log(path):

    pattern = re.compile(
        r"Epoch:\s+(\d+)\s+"
        r"nll_train:\s+([-+0-9.eE]+)\s+"
        r"kl_train:\s+([-+0-9.eE]+)\s+"
        r"mse_train:\s+([-+0-9.eE]+)\s+"
        r"acc_train:\s+([-+0-9.eE]+)\s+"
        r"nll_val:\s+([-+0-9.eE]+)\s+"
        r"kl_val:\s+([-+0-9.eE]+)\s+"
        r"mse_val:\s+([-+0-9.eE]+)\s+"
        r"acc_val:\s+([-+0-9.eE]+)\s+"
        r"time:\s+([-+0-9.eE]+)s"
    )

    data = []

    with open(path, "r") as f:

        for line in f:

            match = pattern.search(line)

            if match:

                values = match.groups()

                data.append({
                    "epoch": int(values[0]),
                    "nll_train": float(values[1]),
                    "kl_train": float(values[2]),
                    "mse_train": float(values[3]),
                    "acc_train": float(values[4]),
                    "nll_val": float(values[5]),
                    "kl_val": float(values[6]),
                    "mse_val": float(values[7]),
                    "acc_val": float(values[8]),
                    "time": float(values[9]),
                })

    return data


# ============================================================
# ディレクトリ作成
# ============================================================

particle_dir = os.path.join(
    OUTPUT_DIR,
    "particle_comparison"
)

data_size_dir = os.path.join(
    OUTPUT_DIR,
    "data_size_comparison"
)

time_dir = os.path.join(
    OUTPUT_DIR,
    "time_comparison"
)

os.makedirs(particle_dir, exist_ok=True)
os.makedirs(data_size_dir, exist_ok=True)
os.makedirs(time_dir, exist_ok=True)


# ============================================================
# データ読み込み
# ============================================================

particle_results = {}
data_size_results = {}


# ------------------------------------------------------------
# 粒子数比較
# ------------------------------------------------------------

print("=" * 70)
print("粒子数比較")
print("=" * 70)

for folder, particles, data_size in PARTICLE_COMPARISON:

    path = (
        f"logs/{folder}/"
        f"sumple{data_size}_100_100/"
        f"log.txt"
    )

    if not os.path.exists(path):

        print(f"[ERROR] ファイルがありません: {path}")
        continue

    data = read_log(path)

    if len(data) == 0:

        print(f"[ERROR] 読み込み失敗: {path}")
        continue

    particle_results[particles] = {
        "data": data,
        "data_size": data_size,
        "path": path,
    }

    times = np.array([
        x["time"] for x in data
    ])

    print(
        f"{particles} particles : "
        f"{len(data)} epochs, "
        f"total = {times.sum():.2f}s, "
        f"mean = {times.mean():.4f}s"
    )


# ------------------------------------------------------------
# 5粒子・データサイズ比較
# ------------------------------------------------------------

print()
print("=" * 70)
print("5粒子・学習データ数比較")
print("=" * 70)

for data_size in DATA_SIZE_COMPARISON:

    path = (
        f"logs/5particle/"
        f"sumple{data_size}_100_100/"
        f"log.txt"
    )

    if not os.path.exists(path):

        print(f"[ERROR] ファイルがありません: {path}")
        continue

    data = read_log(path)

    if len(data) == 0:

        print(f"[ERROR] 読み込み失敗: {path}")
        continue

    data_size_results[data_size] = {
        "data": data,
        "path": path,
    }

    times = np.array([
        x["time"] for x in data
    ])

    print(
        f"data = {data_size:5d} : "
        f"{len(data)} epochs, "
        f"total = {times.sum():.2f}s, "
        f"mean = {times.mean():.4f}s"
    )


# ============================================================
# グラフ作成関数
# ============================================================

def plot_particle_metric(
    metric,
    split,
    filename,
    ylabel,
    title
):

    plt.figure(figsize=(10, 6))

    for particles in sorted(particle_results):

        data = particle_results[particles]["data"]

        epochs = [
            x["epoch"] for x in data
        ]

        values = [
            x[f"{metric}_{split}"]
            for x in data
        ]

        plt.plot(
            epochs,
            values,
            label=f"{particles} particles"
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            particle_dir,
            filename
        ),
        dpi=300
    )

    plt.close()


def plot_data_size_metric(
    metric,
    split,
    filename,
    ylabel,
    title
):

    plt.figure(figsize=(10, 6))

    for data_size in sorted(data_size_results):

        data = data_size_results[data_size]["data"]

        epochs = [
            x["epoch"] for x in data
        ]

        values = [
            x[f"{metric}_{split}"]
            for x in data
        ]

        plt.plot(
            epochs,
            values,
            label=f"{data_size}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(
        title="Training data size"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            data_size_dir,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# ① 粒子数比較グラフ
# ============================================================

# ------------------------
# MSE
# ------------------------

plot_particle_metric(
    "mse",
    "train",
    "mse_train.png",
    "MSE",
    "Training MSE: Particle Count Comparison"
)

plot_particle_metric(
    "mse",
    "val",
    "mse_val.png",
    "MSE",
    "Validation MSE: Particle Count Comparison"
)

# ------------------------
# KL
# ------------------------

plot_particle_metric(
    "kl",
    "train",
    "kl_train.png",
    "KL",
    "Training KL: Particle Count Comparison"
)

plot_particle_metric(
    "kl",
    "val",
    "kl_val.png",
    "KL",
    "Validation KL: Particle Count Comparison"
)

# ------------------------
# ACC
# ------------------------

plot_particle_metric(
    "acc",
    "train",
    "acc_train.png",
    "Accuracy",
    "Training Accuracy: Particle Count Comparison"
)

plot_particle_metric(
    "acc",
    "val",
    "acc_val.png",
    "Accuracy",
    "Validation Accuracy: Particle Count Comparison"
)


# ============================================================
# ② 5粒子・データサイズ比較グラフ
# ============================================================

# ------------------------
# MSE
# ------------------------

plot_data_size_metric(
    "mse",
    "train",
    "mse_train.png",
    "MSE",
    "Training MSE: Training Data Size Comparison"
)

plot_data_size_metric(
    "mse",
    "val",
    "mse_val.png",
    "MSE",
    "Validation MSE: Training Data Size Comparison"
)

# ------------------------
# KL
# ------------------------

plot_data_size_metric(
    "kl",
    "train",
    "kl_train.png",
    "KL",
    "Training KL: Training Data Size Comparison"
)

plot_data_size_metric(
    "kl",
    "val",
    "kl_val.png",
    "KL",
    "Validation KL: Training Data Size Comparison"
)

# ------------------------
# ACC
# ------------------------

plot_data_size_metric(
    "acc",
    "train",
    "acc_train.png",
    "Accuracy",
    "Training Accuracy: Training Data Size Comparison"
)

plot_data_size_metric(
    "acc",
    "val",
    "acc_val.png",
    "Accuracy",
    "Validation Accuracy: Training Data Size Comparison"
)


# ============================================================
# 時間データ
# ============================================================

particle_time = []
data_size_time = []


# ------------------------------------------------------------
# 粒子数
# ------------------------------------------------------------

for particles in sorted(particle_results):

    data = particle_results[particles]["data"]

    times = np.array([
        x["time"] for x in data
    ])

    particle_time.append({
        "particles": particles,
        "total_time_sec": times.sum(),
        "mean_epoch_sec": times.mean(),
    })


# ------------------------------------------------------------
# データサイズ
# ------------------------------------------------------------

for data_size in sorted(data_size_results):

    data = data_size_results[data_size]["data"]

    times = np.array([
        x["time"] for x in data
    ])

    data_size_time.append({
        "data_size": data_size,
        "total_time_sec": times.sum(),
        "mean_epoch_sec": times.mean(),
    })


# ============================================================
# 時間グラフ
# ============================================================

# ------------------------------------------------------------
# 粒子数 × 総時間
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

x = [
    x["particles"]
    for x in particle_time
]

y = [
    x["total_time_sec"] / 60
    for x in particle_time
]

plt.plot(
    x,
    y,
    marker="o"
)

plt.xlabel("Number of particles")
plt.ylabel("Total training time (min)")
plt.title("Total Training Time vs Particle Count")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        time_dir,
        "particle_total_time.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 粒子数 × 1epoch
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

y = [
    x["mean_epoch_sec"]
    for x in particle_time
]

plt.plot(
    x,
    y,
    marker="o"
)

plt.xlabel("Number of particles")
plt.ylabel("Mean time per epoch (s)")
plt.title("Training Time per Epoch vs Particle Count")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        time_dir,
        "particle_epoch_time.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# データサイズ × 総時間
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

x = [
    x["data_size"]
    for x in data_size_time
]

y = [
    x["total_time_sec"] / 60
    for x in data_size_time
]

plt.plot(
    x,
    y,
    marker="o"
)

plt.xlabel("Training data size")
plt.ylabel("Total training time (min)")
plt.title("Total Training Time vs Training Data Size")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        time_dir,
        "data_size_total_time.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# データサイズ × 1epoch
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

y = [
    x["mean_epoch_sec"]
    for x in data_size_time
]

plt.plot(
    x,
    y,
    marker="o"
)

plt.xlabel("Training data size")
plt.ylabel("Mean time per epoch (s)")
plt.title("Training Time per Epoch vs Training Data Size")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        time_dir,
        "data_size_epoch_time.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# summary.csv
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "summary.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "comparison",
        "particles",
        "training_data_size",
        "epochs",
        "total_time_sec",
        "total_time_min",
        "mean_epoch_time_sec",
        "final_train_mse",
        "final_val_mse",
        "final_train_kl",
        "final_val_kl",
        "final_train_acc",
        "final_val_acc",
    ])

    # --------------------------------------------------------
    # 粒子数比較
    # --------------------------------------------------------

    for particles in sorted(particle_results):

        result = particle_results[particles]

        data = result["data"]

        times = np.array([
            x["time"] for x in data
        ])

        final = data[-1]

        writer.writerow([
            "particle_comparison",
            particles,
            result["data_size"],
            len(data),
            times.sum(),
            times.sum() / 60,
            times.mean(),
            final["mse_train"],
            final["mse_val"],
            final["kl_train"],
            final["kl_val"],
            final["acc_train"],
            final["acc_val"],
        ])

    # --------------------------------------------------------
    # データサイズ比較
    # --------------------------------------------------------

    for data_size in sorted(data_size_results):

        result = data_size_results[data_size]

        data = result["data"]

        times = np.array([
            x["time"] for x in data
        ])

        final = data[-1]

        writer.writerow([
            "data_size_comparison",
            5,
            data_size,
            len(data),
            times.sum(),
            times.sum() / 60,
            times.mean(),
            final["mse_train"],
            final["mse_val"],
            final["kl_train"],
            final["kl_val"],
            final["acc_train"],
            final["acc_val"],
        ])


# ============================================================
# 完了
# ============================================================

print()
print("=" * 70)
print("解析完了")
print("=" * 70)

print()
print(f"出力先:")
print(f"  {OUTPUT_DIR}/")

print()
print("生成したグラフ:")
print("  particle_comparison/")
print("  data_size_comparison/")
print("  time_comparison/")

print()
print(f"CSV:")
print(f"  {csv_path}")