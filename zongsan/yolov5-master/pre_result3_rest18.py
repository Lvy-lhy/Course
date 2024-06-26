import subprocess
import shutil
import os
import glob
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# 定义要运行的命令和参数
command = [
    'python', 'detect.py',
    '--source', './Maskdata/val_test1.0/images/',
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

# 清空目标目录中的旧文件
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir)

# 复制最新 crops 目录到目标目录
shutil.copytree(crops_dir, target_dir, dirs_exist_ok=True)

# 复制最新 crops 目录到目标目录
shutil.copytree(crops_dir, target_dir, dirs_exist_ok=True)

print("Crops copied successfully.")

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
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 创建ResNet模型实例
model = ResNetClassifier(num_classes=58)

# 加载模型参数
model.load_state_dict(torch.load("rest8_model/rest18_checkpoint .pth"))
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
    9: "禁止执行和右转",
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
    44: "村庄",
    45: "反向弯路",
    46: "无人看守铁道路口",
    47: "施工",
    48: "连续弯路",
    49: "有人看守铁道路口",
    50: "事故易发路段",
    51: "停车让行",
    52: "禁止通行",
    53: "禁止车辆停放",
    54: "禁止驶入",
    55: "减速让行",
    56: "停车检查",
    57: "停车让行"
}

# 定义预测函数
def predict_image(image_path, model, transform):
    # 读取图像并预处理
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批处理维度

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_label = predicted.item()  # 返回预测结果的标签
    return predicted_label, label_map[predicted_label]  # 返回标签和中文解释

# 遍历文件夹进行预测
base_dir = target_dir
categories = ['mandatory', 'prohibitory', 'warning']

for category in categories:
    category_dir = os.path.join(base_dir, category)
    if os.path.exists(category_dir):
        # 打印正在处理的类别信息
        print(f"Processing category: {category} in folder: {category_dir}")
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                predicted_label, label_explanation = predict_image(image_path, model, transform)
                print(
                    f"Image: {image_name} | Category: {category} | Predicted Label: {predicted_label} | Label Explanation: {label_explanation}")
