import subprocess
import shutil
import os
import glob

# 定义要运行的命令和参数
command = [
    'python', 'detect.py',
    '--source', './Maskdata/train/images/',
    '--weights', './runs/train/exp43/weights/best.pt',
    '--save-crop'
]

# 使用subprocess运行命令
result = subprocess.run(command, capture_output=True, text=True)

# 确定runs/detect下最新的exp_noAdd目录
detect_dirs_pattern = './runs/detect/exp_noAdd*/'
latest_detect_dir = max(glob.glob(detect_dirs_pattern), key=os.path.getmtime)

# 获取最新的 crops 目录
crops_dir = os.path.join(latest_detect_dir, 'crops')

# 目标目录
target_dir = './target_directory/'

# 复制最新 crops 目录到目标目录
shutil.copytree(crops_dir, target_dir, dirs_exist_ok=True)

print("Files copied successfully.")
