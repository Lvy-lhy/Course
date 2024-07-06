import subprocess
import shutil
import os
import glob

# 定义要运行的命令和参数
def run_yolov5_detection(source_dir, weights_path):
    command = [
        'python', 'detect.py',
        '--source', source_dir,
        '--weights', weights_path,
        '--save-crop'
    ]

    # 使用subprocess运行命令
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running detect.py: {result.stderr}")
        return None

    # 确定runs/detect下最新的exp_noAdd目录
    detect_dirs_pattern = './runs/detect/exp_noAdd*/'
    latest_detect_dir = max(glob.glob(detect_dirs_pattern), key=os.path.getmtime)

    # 获取最新的 crops 目录
    crops_dir = os.path.join(latest_detect_dir, 'crops')

    return crops_dir

# 目标目录
target_dir = './target_directory/'

# 定义要运行的命令和参数
source_directory = './Maskdata/val/images/'  # 替换为你希望进行预测的图像目录
yolov5_weights_path = './runs/train/exp43/weights/best.pt'  # 替换为你的YOLOv5模型权重路径

# 运行YOLOv5检测并获取裁剪结果目录
crops_dir = run_yolov5_detection(source_directory, yolov5_weights_path)

if crops_dir:
    # 确保目标目录存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 提取crops目录下的所有图片文件
    image_files = []
    for root, dirs, files in os.walk(crops_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    # 复制所有图片文件到目标目录
    for image_file in image_files:
        shutil.copy(image_file, target_dir)

    print(f"Files from {crops_dir} copied successfully to {target_dir}")
else:
    print("YOLOv5 detection failed. Check logs for details.")
