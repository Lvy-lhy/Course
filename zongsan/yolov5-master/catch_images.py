import subprocess

# 定义要运行的命令和参数
command = [
    'python', 'detect.py',
    '--source', './Maskdata/val/images/',
    '--weights', './runs/train/exp43/weights/best.pt',
    '--save-crop'
]

# 使用subprocess运行命令
result = subprocess.run(command, capture_output=True, text=True)

# 打印输出
print("Standard Output:\n", result.stdout)
print("Standard Error:\n", result.stderr)
