import subprocess
import shutil
import os
import glob
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
from PyQt5 import QtWidgets, QtGui, QtCore

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为模型期望的尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
])

# 使用预训练的ResNet18模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 创建ResNet模型实例
model = ResNetClassifier(num_classes=58)

# 加载模型参数
model.load_state_dict(torch.load("rest18_model/rest18_checkpoint .pth", map_location=torch.device('cpu')))
model.eval()  # 设置模型为评估模式

# 定义类别与中文解释的映射字典
label_map = {
    0: "限速5公里/小时",
    1: "限速15公里/小时",
    2: "限速30公里/小时",
    3: "限速40公里/小时",
    4: "限速50公里/小时",
    5: "限速60公里/小时",
    6: "限速70公里/小时",
    7: "限速80公里/小时",
    8: "禁止直行和左转",
    9: "禁止直行和右转",
    10: "禁止直行",
    11: "禁止左转",
    12: "禁止左右转弯",
    13: "禁止右转",
    14: "禁止超车",
    15: "禁止掉头",
    16: "禁止机动车通行",
    17: "禁止鸣笛",
    18: "解除40限速",
    19: "解除50限速",
    20: "直行或右转",
    21: "直行",
    22: "左转",
    23: "向左和向右转弯",
    24: "向右转",
    25: "靠左侧道路行驶",
    26: "靠右侧道路行驶",
    27: "环岛行驶",
    28: "机动车行驶",
    29: "鸣喇叭",
    30: "非机动车行驶",
    31: "允许掉头",
    32: "左右绕行",
    33: "注意信号灯",
    34: "注意危险",
    35: "注意行人",
    36: "注意非机动车",
    37: "注意儿童",
    38: "向右急转弯",
    39: "向左急转弯",
    40: "下陡坡",
    41: "上陡坡",
    42: "注意慢行",
    43: "T形交叉",
    44: "T形交叉",
    45: "村庄",
    46: "反向弯路",
    47: "无人看守铁道路口",
    48: "施工",
    49: "连续弯路",
    50: "有人看守铁道路口",
    51: "事故易发路段",
    52: "停车让行",
    53: "禁止通行",
    54: "禁止车辆停放",
    55: "禁止驶入",
    56: "减速让行",
    57: "停车检查",
}


class IntermediateImagesWindow(QtWidgets.QWidget):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        self.current_image_index = 0

        self.setWindowTitle('Intermediate Images')
        self.setGeometry(200, 200, 800, 600)

        self.label_image = QtWidgets.QLabel(self)
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setFixedSize(600, 400)

        self.btn_prev = QtWidgets.QPushButton('上一张', self)
        self.btn_prev.setFixedSize(100, 30)
        self.btn_prev.clicked.connect(self.show_prev_image)

        self.btn_next = QtWidgets.QPushButton('下一张', self)
        self.btn_next.setFixedSize(100, 30)
        self.btn_next.clicked.connect(self.show_next_image)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label_image)

        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)

        self.setLayout(layout)
        self.show_image()

    def show_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            pixmap = QtGui.QPixmap(image_path)
            self.label_image.setPixmap(pixmap.scaledToWidth(600))

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_image()
# PyQt5 界面
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.title = '图像分类'
        self.left = 100
        self.top = 100
        self.width = 1400
        self.height = 800
        self.initUI()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)



        # 创建布局
        main_layout = QtWidgets.QVBoxLayout(self)

        # 创建上侧布局，用于显示图片
        image_layout = QtWidgets.QHBoxLayout()

        # 左侧显示裁剪后的图片
        self.cropped_image_label = QtWidgets.QLabel(self)
        self.cropped_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cropped_image_label.setFixedSize(600, 400)
        image_layout.addWidget(self.cropped_image_label)

        # 添加中间图片按钮
        self.btn_intermediate = QtWidgets.QPushButton('中间图片', self)
        self.btn_intermediate.setFixedSize(150, 30)
        self.btn_intermediate.clicked.connect(self.show_intermediate_images)
        m.addWidget(self.btn_intermediate)
        # 右侧显示原始图片
        self.original_image_label = QtWidgets.QLabel(self)
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_image_label.setFixedSize(600, 400)
        image_layout.addWidget(self.original_image_label)

        main_layout.addLayout(image_layout)

        # 创建下侧布局，用于显示预测结果和导航按钮
        bottom_layout = QtWidgets.QVBoxLayout()

        # 文本框，显示预测结果
        self.text_edit = QtWidgets.QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedSize(1200, 200)
        bottom_layout.addWidget(self.text_edit)

        # 导航按钮布局
        nav_layout = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton('上一张', self)
        self.btn_prev.setFixedSize(100, 30)
        self.btn_prev.clicked.connect(self.show_prev_image)
        nav_layout.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton('下一张', self)
        self.btn_next.setFixedSize(100, 30)
        self.btn_next.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.btn_next)

        bottom_layout.addLayout(nav_layout)

        # 将底部布局添加到主布局中
        main_layout.addLayout(bottom_layout)

        # 创建右侧布局，用于放置按钮
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setAlignment(QtCore.Qt.AlignTop)

        # 选择文件夹按钮
        self.btn_select_folder = QtWidgets.QPushButton('选择文件夹', self)
        self.btn_select_folder.setFixedSize(150, 30)
        self.btn_select_folder.clicked.connect(self.select_folder)
        right_layout.addWidget(self.btn_select_folder)

        # 裁剪图片按钮
        self.btn_crop_images = QtWidgets.QPushButton('裁剪图片', self)
        self.btn_crop_images.setFixedSize(150, 30)
        self.btn_crop_images.clicked.connect(self.crop_images)
        right_layout.addWidget(self.btn_crop_images)

        # 预测图片按钮
        self.btn_predict = QtWidgets.QPushButton('预测图片', self)
        self.btn_predict.setFixedSize(150, 30)
        self.btn_predict.clicked.connect(self.predict_images)
        right_layout.addWidget(self.btn_predict)

        # 添加一个占位符，确保按钮靠上对齐
        right_layout.addStretch(1)

        # 将右侧布局添加到主布局中
        main_layout.addLayout(right_layout)

        # 设置主布局
        self.setLayout(main_layout)

        # 初始化变量
        self.source_folder = None
        self.target_dir = './target_directory/'
        self.original_image_map = {}
        self.image_list = []
        self.predictions = []
        self.current_image_index = 0

        # 加载裁剪图片映射（如果存在）
        self.load_image_map()

    def select_folder(self):
        self.source_folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹")
        if self.source_folder:
            print(f"已选择文件夹: {self.source_folder}")

    def crop_images(self):
        if not self.source_folder:
            print("请先选择文件夹")
            return

        # 定义要运行的命令和参数
        command = [
            'python', 'detect.py',
            '--source', self.source_folder,
            '--weights', './runs/train/exp43/weights/best.pt',
            '--save-crop'
        ]

        # 使用subprocess运行命令
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"运行检测脚本时出错: {result.stderr}")
            return

        # 确定runs/detect下最新的exp_noAdd目录
        detect_dirs_pattern = './runs/detect/exp_noAdd*/'
        detect_dirs = glob.glob(detect_dirs_pattern)

        if not detect_dirs:
            print("未找到检测目录.")
            return

        latest_detect_dir = max(detect_dirs, key=os.path.getmtime)

        # 获取最新的 crops 目录
        crops_dir = os.path.join(latest_detect_dir, 'crops')

        # 检查 crops 目录是否存在
        if not os.path.exists(crops_dir):
            print(f"未找到裁剪目录: {crops_dir}")
            return

        # 清空目标目录中的旧文件
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir)

        # 复制最新 crops 目录到目标目录
        shutil.copytree(crops_dir, self.target_dir, dirs_exist_ok=True)

        # 更新裁剪图片映射
        self.update_image_map()

    def update_image_map(self):
        # 清空原始图片映射
        self.original_image_map.clear()

        # 遍历裁剪后的目录结构，建立裁剪后与原始图片的映射关系
        categories = ['mandatory', 'prohibitory', 'warning']

        for category in categories:
            category_dir = os.path.join(self.target_dir, category)
            if os.path.exists(category_dir):
                print(f"处理类别: {category}，文件夹: {category_dir}")
                for cropped_image_name in os.listdir(category_dir):
                    cropped_image_path = os.path.join(category_dir, cropped_image_name)
                    if cropped_image_path.endswith(('.png', '.jpg', '.jpeg')):
                        # 原始图片的路径
                        original_image_name = cropped_image_name.split('_')[0] + '.jpg'  # 假设裁剪前的图片是jpg格式
                        original_image_path = os.path.join(self.source_folder, original_image_name)
                        if os.path.exists(original_image_path):
                            self.original_image_map[cropped_image_path] = original_image_path

    def load_image_map(self):
        map_file = os.path.join(self.target_dir, 'image_map.json')
        if os.path.exists(map_file):
            with open(map_file, 'r') as f:
                self.original_image_map = json.load(f)

    def save_image_map(self):
        map_file = os.path.join(self.target_dir, 'image_map.json')
        with open(map_file, 'w') as f:
            json.dump(self.original_image_map, f)

    def predict_images(self):
        # 清空预测结果和图片列表
        self.predictions.clear()
        self.image_list.clear()

        # 遍历目标目录进行预测
        categories = ['mandatory', 'prohibitory', 'warning']

        for category in categories:
            category_dir = os.path.join(self.target_dir, category)
            if os.path.exists(category_dir):
                print(f"处理类别: {category}，文件夹: {category_dir}")
                for cropped_image_name in os.listdir(category_dir):
                    cropped_image_path = os.path.join(category_dir, cropped_image_name)
                    if cropped_image_path.endswith(('.png', '.jpg', '.jpeg')):
                        # 使用裁剪后的图片进行预测
                        predicted_label, label_explanation = self.predict_image(cropped_image_path)

                        # 找到对应的原始图片路径
                        original_image_path = self.original_image_map.get(cropped_image_path, None)
                        if original_image_path:
                            self.predictions.append({
                                'cropped_image_name': cropped_image_name,
                                'original_image_path': original_image_path,
                                'predicted_label': predicted_label,
                                'label_explanation': label_explanation
                            })
                            self.image_list.append(cropped_image_path)

        if not self.image_list:
            print("目标目录中未找到图片.")
            return

        # 显示第一张图片和预测结果
        self.show_image_and_predictions(self.current_image_index)

    def predict_image(self, image_path):
        # 读取图像并预处理
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # 添加批处理维度

        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        predicted_label = predicted.item()  # 返回预测结果的标签
        label_explanation = label_map.get(predicted_label, "未知类别")  # 返回标签对应的中文解释，未知类别作为默认值
        return predicted_label, label_explanation

    def show_image_and_predictions(self, index):
        if 0 <= index < len(self.image_list):
            cropped_image_path = self.image_list[index]
            original_image_path = self.predictions[index]['original_image_path']

            # 显示裁剪后的图片
            cropped_pixmap = QtGui.QPixmap(cropped_image_path)
            self.cropped_image_label.setPixmap(cropped_pixmap.scaledToWidth(600))

            # 显示原始图片
            original_pixmap = QtGui.QPixmap(original_image_path)
            self.original_image_label.setPixmap(original_pixmap.scaledToWidth(600))

            # 显示预测结果
            prediction = self.predictions[index]
            self.text_edit.clear()
            self.text_edit.append(f"裁剪后图片: {prediction['cropped_image_name']}")
            self.text_edit.append(f"原始图片: {os.path.basename(original_image_path)}")
            self.text_edit.append(f"预测标签: {prediction['predicted_label']}")
            self.text_edit.append(f"标签解释: {prediction['label_explanation']}")
        else:
            self.cropped_image_label.clear()
            self.original_image_label.clear()
            self.text_edit.clear()
            self.text_edit.append("无图片可显示.")

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image_and_predictions(self.current_image_index)

    def show_next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image_and_predictions(self.current_image_index)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
