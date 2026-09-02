import subprocess
import sys


# ============================================================
# 15粒子を実行
# ============================================================

print('\n' + '=' * 60)
print('15 PARTICLE EXPERIMENT START')
print('=' * 60 + '\n')

result = subprocess.run([
    sys.executable,
    'automate_nri_15particle.py'
])

if result.returncode != 0:
    print('\n15粒子の実験でエラーが発生しました。')
    sys.exit(result.returncode)


print('\n' + '=' * 60)
print('15 PARTICLE EXPERIMENT FINISHED')
print('=' * 60 + '\n')


# ============================================================
# 20粒子を実行
# ============================================================

print('\n' + '=' * 60)
print('20 PARTICLE EXPERIMENT START')
print('=' * 60 + '\n')

result = subprocess.run([
    sys.executable,
    'automate_nri_20particle.py'
])

if result.returncode != 0:
    print('\n20粒子の実験でエラーが発生しました。')
    sys.exit(result.returncode)


print('\n' + '=' * 60)
print('ALL EXPERIMENTS FINISHED!')
print('=' * 60 + '\n')