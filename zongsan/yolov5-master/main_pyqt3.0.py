import subprocess
import shutil
import os
import glob
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
    58: "停车让行"
}

# PyQt5 界面
class App(QtWidgets.QWidget):  # 继承自 QtWidgets.QWidget
    def __init__(self):
        super().__init__()
        self.title = '图像分类'
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 700
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 创建布局
        main_layout = QtWidgets.QHBoxLayout(self)

        # 创建左侧布局，用于显示图片和预测结果
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setAlignment(QtCore.Qt.AlignTop)

        # 图片显示标签
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)
        left_layout.addWidget(self.image_label)

        # 文本框，显示预测结果
        self.text_edit = QtWidgets.QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedSize(600, 200)
        left_layout.addWidget(self.text_edit)

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

        left_layout.addLayout(nav_layout)

        # 将左侧布局添加到主布局中
        main_layout.addLayout(left_layout)

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

        # 查看中间过程按钮
        self.btn_view_intermediate = QtWidgets.QPushButton('查看中间过程', self)
        self.btn_view_intermediate.setFixedSize(150, 30)
        self.btn_view_intermediate.clicked.connect(self.view_intermediate)
        right_layout.addWidget(self.btn_view_intermediate)

        # 添加一个占位符，确保按钮靠上对齐
        right_layout.addStretch(1)

        # 将右侧布局添加到主布局中
        main_layout.addLayout(right_layout)

        # 设置主布局
        self.setLayout(main_layout)

        # 初始化变量
        self.source_folder = None
        self.target_dir = None
        self.image_list = []
        self.predictions = []
        self.current_image_index = 0

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

        # 目标目录
        self.target_dir = './target_directory/'

        # 清空目标目录中的旧文件
        if os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)
        os.makedirs(self.target_dir)

        # 复制最新 crops 目录中的所有图片到目标目录
        categories = ['mandatory', 'prohibitory', 'warning']
        for category in categories:
            category_dir = os.path.join(crops_dir, category)
            if os.path.exists(category_dir):
                for image_name in os.listdir(category_dir):
                    image_path = os.path.join(category_dir, image_name)
                    if image_path.endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(image_path, os.path.join(self.target_dir, f"{category}_{image_name}"))

    def predict_images(self):
        if not self.target_dir:
            print("请先裁剪图片")
            return

        # 清空预测结果和图片列表
        self.predictions.clear()
        self.image_list.clear()

        # 遍历目标目录进行预测
        for image_name in os.listdir(self.target_dir):
            image_path = os.path.join(self.target_dir, image_name)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                category = 'unknown'
                if 'mandatory' in image_name:
                    category = 'mandatory'
                elif 'prohibitory' in image_name:
                    category = 'prohibitory'
                elif 'warning' in image_name:
                    category = 'warning'

                predicted_label, label_explanation = self.predict_image(image_path)
                self.predictions.append({
                    'image_name': image_name,
                    'category': category,
                    'predicted_label': predicted_label,
                    'label_explanation': label_explanation
                })
                self.image_list.append(image_path)

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
            image_path = self.image_list[index]
            pixmap = QtGui.QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaledToWidth(600))  # 调整图片宽度显示
            prediction = self.predictions[index]
            self.text_edit.clear()
            self.text_edit.append(f"图片: {prediction['image_name']}")
            self.text_edit.append(f"类别: {prediction['category']}")
            self.text_edit.append(f"预测标签: {prediction['predicted_label']}")
            self.text_edit.append(f"标签解释: {prediction['label_explanation']}")
        else:
            self.image_label.clear()
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

    def view_intermediate(self):
        # 确定runs/detect下最新的目录
        detect_dirs_pattern = './runs/detect/exp*/'
        detect_dirs = glob.glob(detect_dirs_pattern)

        if not detect_dirs:
            print("未找到检测目录.")
            return

        latest_detect_dir = max(detect_dirs, key=os.path.getmtime)

        # 获取最新的目录中的所有图片
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_files.extend(glob.glob(os.path.join(latest_detect_dir, ext)))

        if not image_files:
            print("检测目录中未找到图片.")
            return

        # 显示中间过程图片
        self.intermediate_viewer = IntermediateViewer(image_files)
        self.intermediate_viewer.show()


class IntermediateViewer(QtWidgets.QWidget):
    def __init__(self, image_files):
        super().__init__()
        self.title = '中间过程图片'
        self.left = 150
        self.top = 150
        self.width = 800
        self.height = 700
        self.image_files = image_files
        self.current_image_index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 创建布局
        layout = QtWidgets.QVBoxLayout(self)

        # 图片显示标签
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)
        layout.addWidget(self.image_label)

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

        layout.addLayout(nav_layout)

        # 设置主布局
        self.setLayout(layout)

        # 显示第一张图片
        self.show_image(self.current_image_index)

    def show_image(self, index):
        if 0 <= index < len(self.image_files):
            image_path = self.image_files[index]
            pixmap = QtGui.QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaledToWidth(600))  # 调整图片宽度显示

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
