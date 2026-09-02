import os
import re
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

experiments = {
    'all_sumple100': 'logs/all_sumple100/log.txt',
    'sumple500_100_100': 'logs/sumple500_100_100/log.txt'
}

# 個別グラフの出力先
output_root = 'data_visualization'

# 100 vs 500 の比較グラフの出力先
comparison_root = 'data_visualization/add'


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

            # Epoch行だけを処理
            if not line.startswith('Epoch:'):
                continue

            epoch = int(
                re.search(r'Epoch:\s+(\d+)', line).group(1)
            )

            mse_train_value = float(
                re.search(r'mse_train:\s+([0-9.eE+-]+)', line).group(1)
            )

            kl_train_value = float(
                re.search(r'kl_train:\s+([0-9.eE+-]+)', line).group(1)
            )

            acc_train_value = float(
                re.search(r'acc_train:\s+([0-9.eE+-]+)', line).group(1)
            )

            mse_val_value = float(
                re.search(r'mse_val:\s+([0-9.eE+-]+)', line).group(1)
            )

            kl_val_value = float(
                re.search(r'kl_val:\s+([0-9.eE+-]+)', line).group(1)
            )

            acc_val_value = float(
                re.search(r'acc_val:\s+([0-9.eE+-]+)', line).group(1)
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
# 全実験のログを読み込む
# ============================================================

results = {}

for experiment_name, log_path in experiments.items():

    print(f'Processing: {experiment_name}')

    data = read_log(log_path)

    results[experiment_name] = data

    print(f'読み込んだEpoch数: {len(data[0])}')

    if len(data[0]) == 0:
        print('ERROR: Epochデータを読み込めませんでした')


# ============================================================
# 個別グラフ作成
# ============================================================

for experiment_name, data in results.items():

    (
        epochs,
        mse_train,
        kl_train,
        acc_train,
        mse_val,
        kl_val,
        acc_val
    ) = data

    if len(epochs) == 0:
        continue

    # --------------------------------------------------------
    # 出力フォルダ
    # --------------------------------------------------------

    train_dir = os.path.join(
        output_root,
        experiment_name,
        'train'
    )

    val_dir = os.path.join(
        output_root,
        experiment_name,
        'val'
    )

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # ========================================================
    # TRAIN
    # ========================================================

    # MSE
    plt.figure()
    plt.plot(epochs, mse_train)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{experiment_name} - Train MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'mse.png'))
    plt.close()

    # KL
    plt.figure()
    plt.plot(epochs, kl_train)
    plt.xlabel('Epoch')
    plt.ylabel('KL')
    plt.title(f'{experiment_name} - Train KL')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'kl.png'))
    plt.close()

    # ACC
    plt.figure()
    plt.plot(epochs, acc_train)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{experiment_name} - Train Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'acc.png'))
    plt.close()

    # ========================================================
    # VALIDATION
    # ========================================================

    # MSE
    plt.figure()
    plt.plot(epochs, mse_val)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{experiment_name} - Validation MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, 'mse.png'))
    plt.close()

    # KL
    plt.figure()
    plt.plot(epochs, kl_val)
    plt.xlabel('Epoch')
    plt.ylabel('KL')
    plt.title(f'{experiment_name} - Validation MSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, 'kl.png'))
    plt.close()

    # ACC
    plt.figure()
    plt.plot(epochs, acc_val)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{experiment_name} - Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, 'acc.png'))
    plt.close()

    print('個別グラフ作成完了')


# ============================================================
# 100 vs 500 比較グラフ
# ============================================================

print()
print('100サンプル vs 500サンプル 比較グラフを作成します')


# 出力フォルダ
comparison_train_dir = os.path.join(
    comparison_root,
    'train'
)

comparison_val_dir = os.path.join(
    comparison_root,
    'val'
)

os.makedirs(comparison_train_dir, exist_ok=True)
os.makedirs(comparison_val_dir, exist_ok=True)


# ============================================================
# TRAIN 比較
# ============================================================

# ------------------------------------------------------------
# Train MSE
# ------------------------------------------------------------

plt.figure()

epochs_100 = results['all_sumple100'][0]
mse_train_100 = results['all_sumple100'][1]

epochs_500 = results['sumple500_100_100'][0]
mse_train_500 = results['sumple500_100_100'][1]

plt.plot(
    epochs_100,
    mse_train_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    mse_train_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Train MSE: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_train_dir, 'mse.png')
)

plt.close()


# ------------------------------------------------------------
# Train KL
# ------------------------------------------------------------

plt.figure()

kl_train_100 = results['all_sumple100'][2]
kl_train_500 = results['sumple500_100_100'][2]

plt.plot(
    epochs_100,
    kl_train_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    kl_train_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('KL')
plt.title('Train KL: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_train_dir, 'kl.png')
)

plt.close()


# ------------------------------------------------------------
# Train Accuracy
# ------------------------------------------------------------

plt.figure()

acc_train_100 = results['all_sumple100'][3]
acc_train_500 = results['sumple500_100_100'][3]

plt.plot(
    epochs_100,
    acc_train_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    acc_train_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_train_dir, 'acc.png')
)

plt.close()


# ============================================================
# VALIDATION 比較
# ============================================================

# ------------------------------------------------------------
# Validation MSE
# ------------------------------------------------------------

plt.figure()

mse_val_100 = results['all_sumple100'][4]
mse_val_500 = results['sumple500_100_100'][4]

plt.plot(
    epochs_100,
    mse_val_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    mse_val_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Validation MSE: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_val_dir, 'mse.png')
)

plt.close()


# ------------------------------------------------------------
# Validation KL
# ------------------------------------------------------------

plt.figure()

kl_val_100 = results['all_sumple100'][5]
kl_val_500 = results['sumple500_100_100'][5]

plt.plot(
    epochs_100,
    kl_val_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    kl_val_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('KL')
plt.title('Validation KL: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_val_dir, 'kl.png')
)

plt.close()


# ------------------------------------------------------------
# Validation Accuracy
# ------------------------------------------------------------

plt.figure()

acc_val_100 = results['all_sumple100'][6]
acc_val_500 = results['sumple500_100_100'][6]

plt.plot(
    epochs_100,
    acc_val_100,
    label='100 samples'
)

plt.plot(
    epochs_500,
    acc_val_500,
    label='500 samples'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy: 100 vs 500 samples')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(comparison_val_dir, 'acc.png')
)

plt.close()


# ============================================================
# 完了
# ============================================================

print()
print('すべてのグラフを作成しました。')
print(f'個別グラフ: {output_root}/')
print(f'比較グラフ: {comparison_root}/')