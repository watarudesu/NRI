import os
import sys
import subprocess


# ============================================================
# 設定
# ============================================================

# ============================================================
# 動作確認用
# ============================================================

TRAIN_SIZES = [
    5,
]

NUM_VALID = 100
NUM_TEST = 100
EPOCHS = 500
N_BALLS = 5


# ============================================================
# 1つの実験を実行する関数
# ============================================================

def run_experiment(num_train):

    # ========================================================
    # フォルダ名
    # ========================================================

    experiment_name = (
        f'sumple{num_train}_'
        f'{NUM_VALID}_'
        f'{NUM_TEST}'
    )


    print()
    print()
    print('########################################')
    print(f'EXPERIMENT: {experiment_name}')
    print('########################################')

    print(f'Train   : {num_train}')
    print(f'Valid   : {NUM_VALID}')
    print(f'Test    : {NUM_TEST}')
    print(f'Epochs  : {EPOCHS}')
    print(f'Balls   : {N_BALLS}')


    # ========================================================
    # 1. Dataset generation
    # ========================================================

    print()
    print('========================================')
    print('1. Generating dataset')
    print('========================================')


    generate_script = os.path.join(
        'data',
        'generate_dataset.py'
    )


    result = subprocess.run(
        [
            sys.executable,
            generate_script,

            '--simulation',
            'springs',

            '--num-train',
            str(num_train),

            '--num-valid',
            str(NUM_VALID),

            '--num-test',
            str(NUM_TEST),

            '--n-balls',
            str(N_BALLS),

            '--seed',
            '42'
        ],

        env={
            **os.environ,
            'PYTHONPATH': os.path.abspath('data')
        }
    )


    if result.returncode != 0:

        print()
        print('ERROR')
        print('1. Generating dataset')
        print(f'Return code: {result.returncode}')

        return False


    print()
    print('Dataset generation completed.')


    # ========================================================
    # 2. Training
    # ========================================================

    print()
    print('========================================')
    print('2. Training')
    print('========================================')


    result = subprocess.run(
        [
            sys.executable,
            'train.py',

            '--epochs',
            str(EPOCHS),

            '--num-atoms',
            str(N_BALLS),

            '--data-folder',
            experiment_name,

            '--suffix',
            f'_springs{N_BALLS}',

            '--save-folder',
            'logs',
            
            '--particle-folder',
            '5particle'
        ]
    )


    if result.returncode != 0:

        print()
        print('ERROR')
        print('2. Training')
        print(f'Return code: {result.returncode}')

        return False


    print()
    print('Training completed.')


    # ========================================================
    # 3. Visualization
    # ========================================================

    print()
    print('========================================')
    print('3. Creating graphs')
    print('========================================')


    result = subprocess.run(
        [
            sys.executable,
            'visualize_results.py',

            '--data-folder',
            experiment_name,
            
            '--particle-folder',
            '5particle'
        ]
    )


    if result.returncode != 0:

        print()
        print('ERROR')
        print('3. Visualization')
        print(f'Return code: {result.returncode}')

        return False


    print()
    print('Visualization completed.')


    # ========================================================
    # 完了
    # ========================================================

    print()
    print('========================================')
    print(f'EXPERIMENT COMPLETED: {experiment_name}')
    print('========================================')


    return True


# ============================================================
# 全実験を順番に実行
# ============================================================

print()
print('========================================')
print('NRI AUTOMATION')
print('========================================')

print()
print('Train sizes:')

for size in TRAIN_SIZES:
    print(f'  {size}')


# ============================================================
# 実験ループ
# ============================================================

for num_train in TRAIN_SIZES:

    success = run_experiment(num_train)

    if not success:

        print()
        print('========================================')
        print('AUTOMATION STOPPED')
        print('========================================')

        print(f'Failed experiment: {num_train}')

        sys.exit(1)


# ============================================================
# すべて完了
# ============================================================

print()
print()
print('========================================')
print('ALL EXPERIMENTS COMPLETED')
print('========================================')

print()

for num_train in TRAIN_SIZES:

    experiment_name = (
        f'sumple{num_train}_'
        f'{NUM_VALID}_'
        f'{NUM_TEST}'
    )

    print(f'✓ {experiment_name}')

print()