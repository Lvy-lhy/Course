import subprocess
import os
import glob
import cv2
import matplotlib.pyplot as plt

def run_yolov5_detection(image_path):
    # 构建命令
    command = [
        'python', 'detect.py',
        '--source', image_path,
        '--save-crop'
    ]

    # 使用 subprocess 运行命令
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False

    return True

def display_latest_detection_result(result_dir):
    # 获取最新的检测结果文件
    list_of_files = glob.glob(os.path.join(result_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)

    # 显示最新的检测结果图像
    img = cv2.imread(latest_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # 指定要读取的图片路径
    image_path = '最后测试/14276.jpg'  # 替换为实际图片路径

    # 执行物体检测
    success = run_yolov5_detection(image_path)

    if success:
        # 显示最新的检测结果图像
        display_latest_detection_result('runs/detect')
