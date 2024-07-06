import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import Qt
import subprocess
import shutil
import os
import glob
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageQt
from PyQt5.QtGui import QPixmap

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
        return None, None

    # 确定runs/detect下最新的exp_noAdd目录
    detect_dirs_pattern = './runs/detect/exp_noAdd*/'
    latest_detect_dir = max(glob.glob(detect_dirs_pattern), key=os.path.getmtime)

    # 获取最新的 crops 目录
    crops_dir = os.path.join(latest_detect_dir, 'crops')

    # 获取裁剪前的图片路径
    source_images_dir = os.path.join(latest_detect_dir, 'source')
    list_of_files = glob.glob(os.path.join(source_images_dir, '*.jpg'))  # 假设原始图片是.jpg格式
    latest_source_image = max(list_of_files, key=os.path.getmtime)

    return crops_dir, latest_source_image

# 定义图像转换（与训练时一致）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # 调整大小到224以匹配ResNet输入要求
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
])

# 定义类别标签映射到中文解释的字典
label_map = {
    0: "限速5公里/小时", 1: "限速15公里/小时", 2: "限速30公里/小时", 3: "限速40公里/小时", 4: "限速50公里/小时",
    5: "限速60公里/小时", 6: "限速70公里/小时", 7: "限速80公里/小时", 8: "禁止直行和左转", 9: "禁止直行和右转",
    10: "禁止直行", 11: "禁止左转", 12: "禁止左右转弯", 13: "禁止右转", 14: "禁止超车", 15: "禁止掉头",
    16: "禁止机动车通行", 17: "禁止鸣笛", 18: "解除40限速", 19: "解除50限速", 20: "直行或右转", 21: "直行",
    22: "左转", 23: "向左和向右转弯", 24: "向右转", 25: "靠左侧道路行驶", 26: "靠右侧道路行驶", 27: "环岛行驶",
    28: "机动车行驶", 29: "鸣喇叭", 30: "非机动车行驶", 31: "允许掉头", 32: "左右绕行", 33: "注意信号灯",
    34: "注意危险", 35: "注意行人", 36: "注意非机动车", 37: "注意儿童", 38: "向右急转弯", 39: "向左急转弯",
    40: "下陡坡", 41: "上陡坡", 42: "注意慢行", 43: "T形交叉", 44: "T形交叉", 45: "村庄", 46: "反向弯路",
    47: "无人看守铁道路口", 48: "施工", 49: "连续弯路", 50: "有人看守铁道路口", 51: "事故易发路段", 52: "停车让行",
    53: "禁止通行", 54: "禁止车辆停放", 55: "禁止驶入", 56: "减速让行", 57: "停车检查",
}

# 使用预训练的ResNet模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 创建ResNet模型实例并加载训练好的参数
def load_resnet_model(model_path):
    model = ResNetClassifier(num_classes=58)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置模型为评估模式
    return model

# 定义预测函数
def predict_image(image_path, model, transform, class_names):
    # 读取并处理图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    img = transform(img)
    img = img.unsqueeze(0)  # 添加批次维度

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # 返回预测类别
    return class_names[predicted.item()]

# 批量预测文件夹中的所有图片
def predict_folder(folder_path, model, transform, class_names):
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if
                   fname.endswith(('.jpg', '.jpeg', '.png'))]
    predicted_labels = []
    for image_path in image_paths:
        label = predict_image(image_path, model, transform, class_names)
        predicted_labels.append((image_path, label))
    return predicted_labels

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Traffic Sign Detection and Classification')
        self.setGeometry(100, 100, 1000, 600)

        self.selectFolderBtn = QPushButton('Select Folder', self)
        self.selectFolderBtn.setGeometry(50, 50, 150, 30)
        self.selectFolderBtn.clicked.connect(self.selectFolder)

        self.predictBtn = QPushButton('Predict', self)
        self.predictBtn.setGeometry(250, 50, 100, 30)
        self.predictBtn.clicked.connect(self.predictImages)

        self.sourceImageLabel = QLabel('Source Image:', self)
        self.sourceImageLabel.setGeometry(50, 100, 300, 300)
        self.sourceImageLabel.setAlignment(Qt.AlignCenter)

        self.resultLabel = QLabel('Prediction Results:', self)
        self.resultLabel.setGeometry(400, 100, 500, 450)
        self.resultLabel.setAlignment(Qt.AlignTop)
        self.resultLabel.setWordWrap(True)

    def selectFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.selectedFolder = folder_path
        self.resultLabel.setText(f'Selected Folder: {folder_path}')

    def displayImage(self, image_path, is_source=False):
        pixmap = QPixmap(image_path)
        if is_source:
            scaled_pixmap = pixmap.scaledToHeight(300)
            self.sourceImageLabel.setPixmap(scaled_pixmap)
            self.sourceImageLabel.setAlignment(Qt.AlignCenter)
        else:
            self.resultLabel.setPixmap(pixmap)

    def predictImages(self):
        if hasattr(self, 'selectedFolder'):
            target_dir = './target_directory/'
            crops_dir, source_image_path = run_yolov5_detection(self.selectedFolder, './runs/train/exp43/weights/best.pt')

            if crops_dir and source_image_path:
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                image_files = []
                for root, dirs, files in os.walk(crops_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(os.path.join(root, file))

                for image_file in image_files:
                    shutil.copy(image_file, target_dir)

                model = load_resnet_model('rest18_model/rest18_checkpoint .pth')

                predicted_labels = predict_folder(target_dir, model, transform, label_map)

                result_text = '\n'.join([f'Image: {image_path} | Predicted Label: {label}' for image_path, label in predicted_labels])
                self.resultLabel.setText(f'Prediction Results:\n{result_text}')

                if image_files:
                    # 显示裁剪前的图片
                    self.displayImage(source_image_path, is_source=True)
                    # 显示第一张裁剪后的图片
                    self.displayImage(image_files[0])
            else:
                self.resultLabel.setText("YOLOv5 detection failed. Check logs for details.")

        else:
            self.resultLabel.setText('Please select a folder first.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
