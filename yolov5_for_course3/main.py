import logging
import os
import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

# # 获取当前脚本所在目录的上两级目录，即zongsan/project目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.dirname(current_dir)
# zongsan_dir = os.path.dirname(project_dir)
# print(project_dir)
# sys.path.append(project_dir)
# from zongsan.yolov5_for_course3 import detect_for_course3
import detect_with_add_txt_crop
from Main_ui import Ui_MainWindow
import org_image_inf
import gsmh_rh


# 现在可以使用绝对导入

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.last_position = 0
        self.last_position2 = 0
        self.last_position3 = 0
        self.setupUi(self)
        self.current_index = -1
        self.folder = ""  # 用于存储打开文件的文件夹路径
        # 定义支持的图片和视频格式
        self.image_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        self.video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        self.bt_1.clicked.connect(self.choose_file)  # 选择文件夹
        # self.file_path = r"G:\New_workstations\Course\zongsan\yolov5-master\runs\detect"  # 默认文件路径
        self.file_path = r"G:\New_workstations\Course\zongsan\images_for_detect"  # 最终版本的默认路径
        # self.file_path = r"G:\New_workstations\Course\zongsan\测试图像与识别后图像的匹配功能\runs\detect"
        self.current_file_path = ""  # 存储选择文件夹后，该文件夹的路径
        self.current_image_path = ""  # 存储当前显示的图像的路径
        self.crops_save_path = ""  # 用于存储预测后的crops文件路径
        self.files_path = []  # 用于存储文件夹下的所有文件路径
        self.bt_2.clicked.connect(self.detect)  # 开始检测(监测后，应该自动切换为处理后的文件路径，这样以便与直接切换图片
        self.bt_3.clicked.connect(self.show_next_image)  # 实现切换下一张照片
        self.bt_4.clicked.connect(self.show_last_image)  # 实现切换上一张照片
        self.bt_5.clicked.connect(self.show_last_dealed_image)
        self.bt_6.clicked.connect(self.show_next_dealed_image)
        self.current_sort = [" "]
        self.bt_7.clicked.connect(self.exit_sys)  # 退出程序
        self.bt_9.clicked.connect(self.org_images) # 神经网络识别
        self.current_count = 0  # 用来存储
        # self.inf_1.setPlaceholderText("...")
        self.matching_files = []  # 用于存储当前文件中，识别到的图像路径，注意在切换图像时，要清空其内容
        # self.matching_files_kinds = [] #
        self.i = 0  # 用于记录matching_files当前下标

    def org_images(self):

        return 0

    # def get_crossroad_description(self, txt_path, file_name):
    #     print("测试get_crossroad_description方法是否成功调用")
    #     crossroads_file = txt_path  # 替换为实际的文件路径
    #     with open(crossroads_file, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #     # 根据文件名中的编号查找对应的交叉路口描述
    #     for line in lines:
    #         parts = line.strip().split(': ')
    #         if len(parts) == 2:
    #             num_str, description = parts
    #             if num_str.strip() == file_name:
    #                 self.inf_2.clear()
    #                 self.inf_2.append(description)
    #             else:
    #                 self.inf_2.clear()
    #                 self.inf_2.append("图像库缺失该匹配内容")
    #         else:
    #             print("文件行错误")
    #     return None  # 如果没有找到对应编号的描述，返回 None 或者其他适当的值

    def get_crossroad_description(self, txt_path, file_name):
        try:
            crossroads_file = txt_path  # 替换为实际的文件路径
            with open(crossroads_file, 'r', encoding='gbk') as f:
                lines = f.readlines()
            found_description = False
            # 根据文件名中的编号查找对应的交叉路口描述
            for line in lines:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    num_str, description = parts
                    if num_str.strip() == file_name:
                        self.inf_2.clear()
                        self.inf_2.append(description)
                        found_description = True
                        break  # 找到匹配项后直接退出循环
            if not found_description:
                self.inf_2.clear()
                self.inf_2.append("图像库缺失该匹配内容")
        except FileNotFoundError:
            print(f"文件未找到: ")
        except Exception as e:
            print(f"处理文件时出现错误: {e}")
        return None  # 如果没有找到对应编号的描述，返回 None 或者其他适当的值

    def image_recognition(self, image_path, image_data_path):  # 图片路径和与之匹配的图像库路径
        best_match_image = org_image_inf.find_best_match(image_path, image_data_path)
        print("------------最佳匹配图像路径:test: = " + best_match_image)
        parent_directory = os.path.dirname(best_match_image)  # 获取上一级目录路径
        txt_path = os.path.join(parent_directory, "images_inf.txt")
        image_name = os.path.basename(best_match_image) # 获取文件名
        file_name, file_extension = os.path.splitext(image_name)  # 获取文件名，以通过文件名匹配具体含义
        self.get_crossroad_description(txt_path, file_name) # 传入txt库、与之匹配的图像名
        # print("传入txt库、与之匹配的图像名")
        # print(txt_path)
        # print(file_name)
        return 0

    def get_detail_inf(self, sort, image_path):
        if sort == "指示标志":
            # print("sort = " + sort)
            self.image_recognition(image_path,
                                   r"G:\New_workstations\Course\zongsan\300x300_images_data\number_inf\ZS_300x300")
        elif sort == "禁令标志":
            # print("sort = " + sort)
            self.image_recognition(image_path,
                                   r"G:\New_workstations\Course\zongsan\300x300_images_data\number_inf\JL_300x300")
        elif sort == "警告标志":
            # print("sort = " + sort)
            self.image_recognition(image_path,
                                   r"G:\New_workstations\Course\zongsan\300x300_images_data\number_inf\JG_300x300")
        else:
            print("无法识别当前类型")
            self.inf_2.clear()
            self.inf_2.append("无法识别当前类型")

    def detect(self):
        # detect_path = r"G:\New_workstations\Course\zongsan\yolov5-master"
        # sys.path.append(detect_path)
        # import detect_for_course3
        print(self.current_file_path)
        self.inf_3.append(self.current_file_path)
        self.crops_save_path = detect_with_add_txt_crop.main(self.current_file_path)
        self.inf_3.append("分类后的图像在：" + str(os.path.dirname(self.crops_save_path)) + "路径下")  # 接收该路径
        self.inf_3.moveCursor(QtGui.QTextCursor.End)  # 自动滚动到最新消息

        # print("测试裁切后的文件存储位置")
        # print(self.crops_save_path)

    # def show_crops_images(self): # 显示处理后的图像的裁切信息
    # def show_dealed_images(self):
    #     print("test")

    def show_next_dealed_image(self):
        if self.file_exit:
            self.i += 1
            if self.i < len(self.matching_files):
                self.inf_1.clear()
                self.inf_1.append(self.current_sort[self.i])  # 同步类型切换

                image_path = self.matching_files[self.i]  # 子图像路径
                self.video_2.setPixmap(QPixmap(image_path).scaled(self.video_2.size(), aspectRatioMode=1))  # 新的显示方法
                self.inf_3.moveCursor(QtGui.QTextCursor.End)  # 自动滚动到最新消息
                self.get_detail_inf(self.current_sort[self.i], image_path)  # 模板匹配
                print("当前传入的图像类型和路径")
                print(self.current_sort[self.i])
                print(image_path)
            else:
                self.inf_3.append("已经显示到了最后一张图片")
                self.i = len(self.matching_files) - 1
        else:
            self.inf_1.clear()
            self.inf_1.append("当前图像为检测出交通标识")

    def show_last_dealed_image(self):
        if self.file_exit:
            self.i -= 1
            if self.i < 0:
                self.inf_3.append("当前显示的是第一张图片")
                self.i = 0
            else:


                self.inf_1.clear()
                self.inf_1.append(self.current_sort[self.i])  # 同步类型切换
                image_path = self.matching_files[self.i] # 图片路径

                self.video_2.setPixmap(QPixmap(image_path).scaled(self.video_2.size(), aspectRatioMode=1))  # 新的显示方法
                self.inf_3.moveCursor(QtGui.QTextCursor.End)  # 自动滚动到最新消息

                self.get_detail_inf(self.current_sort[self.i], image_path)  # 传入种类以及图片路径
                print("当前传入的图像类型和路径")
                print(self.current_sort[self.i])
                print(image_path)

        else:
            self.inf_1.clear()
            self.inf_1.append("当前图像未检测出交通标识")

    def get_inf_from_txt(self):  # 从txt文件中，获取识别后图像的框（包含类别、坐标信息）
        self.i = 0
        self.matching_files.clear()  # 每次切换了主图后，获取匹配的分割图像路径需要重新加载记录
        self.current_sort.clear()
        self.last_position = 0
        self.last_position2 = 0
        self.last_position3 = 0

        print(self.current_sort)
        # print("当前图片路径："+ self.current_image_path)
        parent_directory = os.path.dirname(self.current_image_path)  # 上一级目录路径
        # print("当前图片的上一级目录 " + parent_directory)
        last_level_name = os.path.basename(self.current_image_path)  # 即文件名
        # print("文件名 " + last_level_name)
        labels_directory = os.path.join(parent_directory, 'labels')  # labels 目录路径
        # 这里的exp要以实际路径为准，因此需要添加程序更新路径。
        # print("添加路径 " + labels_directory)
        label_file_path = os.path.join(labels_directory, last_level_name)  # 完整的标签文件路径
        # print("标签路径 " + label_file_path)
        label_filename = os.path.splitext(label_file_path)[0] + '.txt'  # 构建对应的标签文件名，例如 '19000.txt'
        # print("最终路径 " + label_filename)
        # print("测试是否找到txt文件：" + label_filename)
        # 检查文件是否存在，并读取内容
        if os.path.exists(label_filename):
            with open(label_filename, 'r') as f:
                file_content = f.readlines()
                self.file_exit = 1
        else:
            print(f"未找到指定路径: {label_filename}")
            self.file_exit = 0

        # 当txt文件存在，说明识别到了相应标志，这个时候才能进行读取文件操作，以及后续的子图切换操作
        if self.file_exit:
            self.first_char = [line[0] for line in file_content]  # 获取txt文件中每一行的首字符（即识别到的类别）
            self.current_count = len(self.first_char)  # 相当于获取有多少个选框。
            # unique_list = list(set(self.first_char))
            # print("测试首字符输出")
            # print(unique_list)
            # self.current_count = len(unique_list)
            # 需要去除重复值，以免重复读取录入路径信息。
            # print("测试首字符输出")
            # print(self.first_char)

            # for i in range(0, self.current_count):
            #     if self.first_char[i] == '0':
            #         dealed_image_path = os.path.join(parent_directory, 'crops\prohibitory')  # 补充目录路径
            #         for filename in os.listdir(dealed_image_path):
            #             # if "_" in filename:
            #             prefix = filename.split("_")[0]
            #             if prefix == os.path.splitext(last_level_name)[0]: # 文件名一样
            #                 self.matching_files.append(os.path.join(dealed_image_path, filename))  # 将匹配到的文件路径加进去
            #                 # print(self.matching_files) # 测试识别文件的路径（成功
            #         self.current_sort = "Z_S_BZ_50x50"
            #     elif self.first_char[i] == '1':
            #         dealed_image_path = os.path.join(parent_directory, 'crops\warning')  # labels 目录路径
            #         for filename in os.listdir(dealed_image_path):
            #             if "_" in filename:
            #                 prefix = filename.split("_")[0]
            #                 if prefix == os.path.splitext(last_level_name)[0]:
            #                     self.matching_files.append(os.path.join(dealed_image_path, filename))
            #
            #         self.current_sort = "禁令标志"
            #     elif self.first_char[i] == '2':
            #         dealed_image_path = os.path.join(parent_directory, 'crops\mandatory')  # labels 目录路径
            #         for filename in os.listdir(dealed_image_path):
            #             if "_" in filename:
            #                 prefix = filename.split("_")[0]
            #                 if prefix == os.path.splitext(last_level_name)[0]:
            #                     self.matching_files.append(os.path.join(dealed_image_path, filename))
            #
            #         self.current_sort = "警告标志"
            #     else:
            #         self.current_sort = "未识别出类型"

            for i in range(0, self.current_count):
                if self.first_char[i] == '0':
                    dealed_image_path = os.path.join(parent_directory, 'crops', 'prohibitory')  # 补充目录路径
                    files = os.listdir(dealed_image_path)  # 获取目录下的文件列表
                    # print("files1:")
                    # print(files1)
                    for j in range(self.last_position, len(files)):
                        # if "_" in filename:
                        filename = files[j]
                        prefix = filename.split("_")[0]
                        if prefix == os.path.splitext(last_level_name)[0]:
                            # 记录匹配的文件
                            self.matching_files.append(os.path.join(dealed_image_path, filename))
                            # self.current_sort.append("Z_S_BZ_50x50")
                            # 更新上次查找的位置
                            self.last_position = j + 1
                            self.current_sort.append("指示标志")  # 在切换图片，程序闪退，添加测试发现数组缺少元素添加
                            break
                            # 如果没有找到匹配文件，重置last_position
                        self.last_position = 0

                elif self.first_char[i] == '1':
                    dealed_image_path = os.path.join(parent_directory, 'crops', 'warning')  # labels 目录路径
                    files = os.listdir(dealed_image_path)  # 获取目录下的文件列表
                    for k in range(self.last_position2, len(files)):
                        # if "_" in filename:
                        filename = files[k]
                        prefix = filename.split("_")[0]
                        if prefix == os.path.splitext(last_level_name)[0]:
                            # 记录匹配的文件
                            self.matching_files.append(os.path.join(dealed_image_path, filename))
                            # 更新上次查找的位置
                            self.last_position2 = k + 1
                            self.current_sort.append("禁令标志")
                            break
                        self.last_position2 = 0

                elif self.first_char[i] == '2':
                    dealed_image_path = os.path.join(parent_directory, 'crops',
                                                     'mandatory')  # labels 目录路径                    files =
                    # os.listdir(dealed_image_path) # 获取目录下的文件列表
                    files = os.listdir(dealed_image_path)  # 获取目录下的文件列表
                    for l in range(self.last_position3, len(files)):
                        # if "_" in filename:
                        filename = files[l]
                        prefix = filename.split("_")[0]
                        if prefix == os.path.splitext(last_level_name)[0]:
                            # 记录匹配的文件
                            self.matching_files.append(os.path.join(dealed_image_path, filename))
                            # 更新上次查找的位置
                            self.last_position3 = l + 1
                            self.current_sort.append("警告标志")
                            break
                        self.last_position3 = 0
                else:
                    self.inf_1.clear()
                    self.inf_1.append("未识别出类型")

            # print("输出当前识别到的类别self.current_sort")
            # print(self.current_sort)
            # 清空内容再显示类别
            # self.inf_1.clear()
            # print("匹配到的文件路径")
            # print(self.matching_files)
            # print("匹配到的文件个数：" + str(len(self.matching_files)))

            # if self.current_count is not None: # 不为空，则说明有匹配到文件，显示第一份
            #     self.inf_1.append(self.current_sort[0])
            # else:
            #     self.inf_1.append("当前图图片未检测出标志")
            # print("self.matching_files")
            # print(self.matching_files)
            print("---test---")
            print("种类:")
            print(self.current_sort)
            print("种类编号：")
            print(self.first_char)
            self.inf_1.clear()

            if self.current_sort:  # 打开文件夹默认显示第一张图片的信息
                self.inf_1.append(self.current_sort[0])  # 显示类别
            else:
                self.inf_1.append("未识别出类别")  # 显示类别

            if self.matching_files:
                image_path = self.matching_files[0]
                self.get_detail_inf(self.current_sort[0], image_path)  # 显示图像的具体含义
                print("当前传入的图像类型和路径")
                print(self.current_sort[0])
                print(image_path)
                self.video_2.setPixmap(QPixmap(image_path).scaled(self.video_2.size(), aspectRatioMode=1))  # 显示图像
                self.inf_3.moveCursor(QtGui.QTextCursor.End)
            else:
                self.inf_3.append("未检测到图像")
        else:
            self.video_2.clear()
            self.inf_3.append("当前图片未检测到交通标志牌")  # 还需要设置，不能实现子图像的切换控制

        print(self.current_image_path)
        print(self.matching_files)
        print(self.current_index)

    def exit_sys(self):
        self.video.setPixmap(QPixmap())  # 清空显示的图像
        sys.exit(0)

    def show_last_image(self):  # 实现切换上一张照片
        if self.files_path:
            self.current_index = (self.current_index - 1) % len(self.files_path)  # 索引值加-1
            image_path = self.files_path[self.current_index]  # 获取第一张图像的路径
            self.video.setPixmap(QPixmap(image_path).scaled(self.video.size(), aspectRatioMode=1))  # 新的显示方法
            self.inf_3.moveCursor(QtGui.QTextCursor.End)
            # 根据 QLabel 的大小缩放图片，并保持图片的宽高比。
            self.current_image_path = image_path
            self.inf_3.append("当前图片路径" + image_path)
            # 调用显示细节信息
            self.get_inf_from_txt()
        else:
            self.inf_3("路径空，图像切换失败")

    def show_next_image(self):  # 实现切换下一张照片
        if self.files_path:
            self.current_index = (self.current_index + 1) % len(self.files_path)  # 索引值加1
            image_path = self.files_path[self.current_index]  # 下一张图片路径
            # self.video.clear()
            self.video.setPixmap(QPixmap(image_path).scaled(self.video.size(), aspectRatioMode=1))  # 新的显示方法
            self.inf_3.moveCursor(QtGui.QTextCursor.End)
            # 根据 QLabel 的大小缩放图片，并保持图片的宽高比。
            self.current_image_path = image_path
            self.inf_3.append("当前图片路径" + image_path)
            # 调用显示细节信息
            self.get_inf_from_txt()
        else:
            self.inf_3("路径空，图像切换失败")

    def get_images_path(self, current_file_path):  # 加载各图像的路径
        self.files_path = [os.path.join(current_file_path, f) for f in os.listdir(current_file_path)
                           if f.lower().endswith(self.image_formats)]
        # self.files_path.sort() # 没什么必要，已经默认排序好了
        self.current_index = -1 if not self.files_path else 0  # 若有图片，则从第一张开始加载。

    def choose_file(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", self.file_path)  # 设置默认路径
        if folder == "":  # 设置默认路径，防止读入空路径导致程序闪退
            folder = r"G:\New_workstations\Course\zongsan\images_t\images_for_test"
        self.current_file_path = folder  # 存储文件夹的路径
        self.get_images_path(self.current_file_path)  # 加载文件夹下的各图片路径
        # print(self.files_path)
        # 选择要识别的文件夹
        # print(folder)
        # 检查文件夹内文件格式
        # logging.info("打开文件: %s", folder)
        # image_name = self.check_file_format(folder)  # 返回当前文件内容
        # image_path = os.path.join(folder,image_name)
        # print(image_path)# 测试路径输出
        # 返回改图像的路径信息
        if self.files_path:  # 文件夹路径
            image_path = self.files_path[self.current_index]  # 获取第一张图像的路径
            print(image_path)
            self.video.setPixmap(QPixmap(image_path).scaled(self.video.size(), aspectRatioMode=1))
            self.inf_3.moveCursor(QtGui.QTextCursor.End)
            # 根据 QLabel 的大小缩放图片，并保持图片的宽高比。
            self.current_image_path = image_path
            self.inf_3.append(f"当前图像路径: {image_path}")
        else:
            self.inf_3.append("未找到图像文件")
        # 还需要检查输入为单个视频或者单张图像的情况…………待补充！！！！！！！！！！！！！！！
        # 在窗口中显示当前图像。
        # self.video.setPixmap(QPixmap(image_path))
        # self.video.setScaledContents(True)  # 设置 QLabel 自适应内容大小
        # self.video.setPixmap(QPixmap(image_path).scaled(self.video.size(), aspectRatioMode=1)) #新的显示方法
        # 根据 QLabel 的大小缩放图片，并保持图片的宽高比。

        # 调用函数显示子图像信息
        self.get_inf_from_txt()

    # 检查文件格式并返回第一个符合要求的文件路径
    def check_file_format(self, folder):
        # self.inf_3.append()
        files = os.listdir(folder)
        if not files:
            self.inf_3.append("文件夹为空")
            logging.info("文件夹为空")
            return
        # 找到第一个图片文件
        first_image = None
        for file in files:
            if file.lower().endswith(self.image_formats):  # 避免文件名后缀大小写不同造成识别差异
                first_image = file
                break
        return first_image


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  #自适应窗口大小
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
