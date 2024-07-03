# import cv2
# import numpy as np
#
#
# def preprocess_image(image_path, num_gaussian_blurs=5):
#     # 读取图像
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if image is None:
#         print(f"无法读取图像: {image_path}")
#         return None
#
#     # 转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 多次应用高斯滤波降噪
#     denoised_image = gray_image
#     for _ in range(num_gaussian_blurs):
#         denoised_image = cv2.GaussianBlur(denoised_image, (5, 5), 20)
#
#     # 应用CLAHE（自适应直方图均衡化）
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(denoised_image)
#
#     # 图像锐化，调整锐化核
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5.6, -1],
#                        [0, -1, 0]])
#     sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)
#
#     return sharpened_image
#
#
# def main(image_path):
#     # image_path = r'G:\New_workstations\Course\zongsan\Detected_images\exp_test17\crops\warning\22_0.jpg'  # 替换为实际图像路径
#     preprocessed_image = preprocess_image(image_path, num_gaussian_blurs=3)  # 设置高斯滤波处理次数
#     if preprocessed_image is not None:
#         return preprocessed_image
#     #     # 显示处理后的图像
#     #     cv2.imshow('Original Image', cv2.imread(image_path))
#     #     cv2.imshow('Preprocessed Image', preprocessed_image)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
#
#
# # if __name__ == "__main__":
# #     main(image_path)
import cv2

def main(img_path, target_size=(300, 300)):
    # 读取原始图片
    img = cv2.imread(img_path)

    # 获取原始图片尺寸
    h, w = img.shape[:2]

    # 计算缩放比例
    if h > w:
        scale_factor = target_size[0] / h
    else:
        scale_factor = target_size[1] / w

    # 缩放图片
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    return resized_img

# # 示例用法
# img_path =r'G:\New_workstations\Course\zongsan\Detected_images\exp_test17\crops\warning\22_0.jpg'
# resized_img = resize_image(img_path)
#
# # 显示缩放后的图片
# cv2.imshow('Resized Image', resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

