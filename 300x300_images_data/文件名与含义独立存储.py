import os
import shutil


def rename_files_in_directory(source_dir, target_dir, log_file):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 确保日志文件的父目录存在
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 打开日志文件用于写入
    with open(log_file, 'w') as log:
        # 获取目录下的所有文件
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        files.sort()  # 对源文件名进行排序

        # 遍历所有文件并重命名
        for index, filename in enumerate(files):
            name, ext = os.path.splitext(filename)  # 返回名字 + 类型
            new_filename = f"{index}{ext}"  # 保留原文件的扩展名，编号+类型
            source_path = os.path.join(source_dir, filename)  # 原路径 + 文件名字
            target_path = os.path.join(target_dir, new_filename)  # 目标路径 + 新的文件名

            shutil.copy(source_path, target_path)  # 复制并重命名文件

            # 写入编号与原文件名（不包括后缀）
            log.write(f"{index}: {name}\n")
            print(f"文件 '{filename}' 被重命名为 '{new_filename}' 并保存到 '{target_dir}'")


if __name__ == "__main__":
    source_directory = r"G:\New_workstations\Course\zongsan\300x300_test\ZS300x300"  # 替换为源目录的路径
    target_directory = r"G:\New_workstations\Course\zongsan\300x300_test\number_inf\ZS_300x300"  # 替换为目标目录的路径
    log_file_path = (r"G:\New_workstations\Course\zongsan\300x300_test\number_inf\ZS_300x300\images_inf"
                     r".txt")  # 替换为日志文件的路径

    rename_files_in_directory(source_directory, target_directory, log_file_path)
