import os
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision import models

# 定义图像转换（与训练时一致）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # 调整大小到224以匹配ResNet输入要求
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正则化
])

# 定义类别标签映射到中文解释的字典
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

# 使用预训练的ResNet模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 创建ResNet模型实例并加载训练好的参数
model = ResNetClassifier(num_classes=58)
model.load_state_dict(torch.load('rest18_model/rest18_checkpoint .pth', map_location=torch.device('cpu')))
model.eval()  # 设置模型为评估模式

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

# 预测文件夹中的所有图片
folder_path = 'target_directory'  # 替换为实际的文件夹路径
predicted_labels = predict_folder(folder_path, model, transform, label_map)

# 打印每张图片的预测结果
for image_path, label in predicted_labels:
    print(f'Image: {image_path} | Predicted Label: {label}')
