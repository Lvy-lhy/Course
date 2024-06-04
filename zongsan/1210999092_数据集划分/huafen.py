import os
import shutil
import random
from tqdm import tqdm

"""
标注文件是yolo格式（txt文件）
训练集：验证集：测试集 （7：2：1） 
"""


def split_img(img_path, label_path, split_list):
    try:
        Data = './Maskdata'
        # 创建需要的文件夹
        train_img_dir = os.path.join(Data, 'train', 'images')
        val_img_dir = os.path.join(Data, 'val', 'images')
        test_img_dir = os.path.join(Data, 'test', 'images')

        train_label_dir = os.path.join(Data, 'train', 'labels')
        val_label_dir = os.path.join(Data, 'val', 'labels')
        test_label_dir = os.path.join(Data, 'test', 'labels')

        # 创建文件夹
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)
        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(test_label_dir, exist_ok=True)

    except Exception as e:
        print(f'文件目录已存在: {e}')

    train, val, test = split_list
    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img]

    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    for img in train_img:
        all_img_path.remove(img)

    val_img = random.sample(all_img_path, int(val / (val + test) * len(all_img_path)))
    for img in val_img:
        all_img_path.remove(img)

    test_img = all_img_path

    # 将图片和标签文件复制到相应的目录
    copy_files(train_img, label_path, train_img_dir, train_label_dir, 'train')
    copy_files(val_img, label_path, val_img_dir, val_label_dir, 'val')
    copy_files(test_img, label_path, test_img_dir, test_label_dir, 'test')


def copy_files(img_list, label_path, img_dir, label_dir, desc):
    for img in tqdm(img_list, desc=f'{desc} ', ncols=80, unit='img'):
        img_name = os.path.basename(img)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        shutil.copy(img, img_dir)
        shutil.copy(os.path.join(label_path, label_name), label_dir)


if __name__ == '__main__':
    img_path = 'JPEGImage'  # 你的图片存放的路径（路径一定是相对于你当前的这个脚本文件而言的）
    label_path = 'Annotations'  # 你的txt文件存放的路径（路径一定是相对于你当前的这个脚本文件而言的）
    split_list = [0.8, 0.2, 0]  # 数据集划分比例[train:val:test]
    split_img(img_path, label_path, split_list)