import os
import re
import argparse
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-folder',
    type=str,
    required=True,
    help='Experiment folder name.'
)

parser.add_argument(
    '--particle-folder',
    type=str,
    required=True,
    help='Particle folder name.'
)

args = parser.parse_args()

experiment_name = args.data_folder
particle_folder = args.particle_folder

log_path = os.path.join(
    'logs',
    particle_folder,
    experiment_name,
    'log.txt'
)

output_root = 'data_visualization'

# ============================================================
# log.txt を読み込む
# ============================================================

def read_log(log_path):

    epochs = []

    mse_train = []
    kl_train = []
    acc_train = []

    mse_val = []
    kl_val = []
    acc_val = []

    with open(log_path, 'r') as f:

        for line in f:

            if not line.startswith('Epoch:'):
                continue

            epoch = int(
                re.search(
                    r'Epoch:\s+(\d+)',
                    line
                ).group(1)
            )

            mse_train_value = float(
                re.search(
                    r'mse_train:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            kl_train_value = float(
                re.search(
                    r'kl_train:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            acc_train_value = float(
                re.search(
                    r'acc_train:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            mse_val_value = float(
                re.search(
                    r'mse_val:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            kl_val_value = float(
                re.search(
                    r'kl_val:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            acc_val_value = float(
                re.search(
                    r'acc_val:\s+([0-9.eE+-]+)',
                    line
                ).group(1)
            )

            epochs.append(epoch)

            mse_train.append(mse_train_value)
            kl_train.append(kl_train_value)
            acc_train.append(acc_train_value)

            mse_val.append(mse_val_value)
            kl_val.append(kl_val_value)
            acc_val.append(acc_val_value)

    return (
        epochs,
        mse_train,
        kl_train,
        acc_train,
        mse_val,
        kl_val,
        acc_val
    )


# ============================================================
# log.txt 読み込み
# ============================================================

print(f'Processing: {experiment_name}')

(
    epochs,
    mse_train,
    kl_train,
    acc_train,
    mse_val,
    kl_val,
    acc_val
) = read_log(log_path)

print(f'読み込んだEpoch数: {len(epochs)}')

if len(epochs) == 0:

    print('ERROR: Epochデータを読み込めませんでした')
    exit()


# ============================================================
# 出力フォルダ
# ============================================================

train_dir = os.path.join(
    output_root,
    particle_folder,
    experiment_name,
    'train'
)

val_dir = os.path.join(
    output_root,
    particle_folder,
    experiment_name,
    'val'
)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


# ============================================================
# TRAIN
# ============================================================

# MSE
plt.figure()
plt.plot(epochs, mse_train)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title(f'{experiment_name} - Train MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(train_dir, 'mse.png')
)
plt.close()


# KL
plt.figure()
plt.plot(epochs, kl_train)
plt.xlabel('Epoch')
plt.ylabel('KL')
plt.title(f'{experiment_name} - Train KL')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(train_dir, 'kl.png')
)
plt.close()


# Accuracy
plt.figure()
plt.plot(epochs, acc_train)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{experiment_name} - Train Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(train_dir, 'acc.png')
)
plt.close()


# ============================================================
# VALIDATION
# ============================================================

# MSE
plt.figure()
plt.plot(epochs, mse_val)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title(f'{experiment_name} - Validation MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(val_dir, 'mse.png')
)
plt.close()


# KL
plt.figure()
plt.plot(epochs, kl_val)
plt.xlabel('Epoch')
plt.ylabel('KL')
plt.title(f'{experiment_name} - Validation KL')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(val_dir, 'kl.png')
)
plt.close()


# Accuracy
plt.figure()
plt.plot(epochs, acc_val)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'{experiment_name} - Validation Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(val_dir, 'acc.png')
)
plt.close()


print('グラフ作成完了')
print(f'出力先: {output_root}/{experiment_name}/')