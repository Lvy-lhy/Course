# import cv2
# import os
# import concurrent.futures
# import numpy as np
# import gsmh_rh
#
#
# # 返回最佳匹配文件的名称，然后可以根据文件名编号，从该路径的txt文件中找出对应的含义。
# def deal_org_image(image):
#     return gsmh_rh.main(image)
#     # return cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#
#
# # 创建SIFT特征检测器
# sift = cv2.SIFT_create(nfeatures=30, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=20, sigma=0.5)
#
# # 创建暴力匹配器
# bf = cv2.BFMatcher()
#
#
# # 定义匹配函数
# def match_image(target_img, image_path):
#     # 读取待匹配图像（灰度图像）
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # 检测和计算SIFT特征
#     kp1, des1 = sift.detectAndCompute(target_img, None)
#     kp2, des2 = sift.detectAndCompute(img, None)
#
#     # 使用暴力匹配器进行特征点匹配
#     matches = bf.knnMatch(des1, des2, k=2)
#
#     # 应用比例测试来筛选好的匹配点
#     good_matches = []
#     for match in matches:
#         if len(match) == 2:  # 确保每个匹配元素都是一个长度为2的列表
#             m, n = match
#             if m.distance < 0.75 * n.distance:
#                 good_matches.append(m)
#
#     # 确保匹配点对的数量足够用于RANSAC
#     if len(good_matches) >= 4:  # 至少需要4对匹配点
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
#         # 使用RANSAC算法进行匹配点筛选
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         matches_mask = mask.ravel().tolist()
#
#         # 计算有效匹配点数
#         num_inliers = np.sum(matches_mask)
#
#     else:
#         matches_mask = None
#         num_inliers = 0
#
#     return image_path, num_inliers, kp1, kp2, good_matches, matches_mask
#
#
# # 定义图像搜索路径
#
# def main(image_path, image_data_path):
#     # 设置目标图像路径
#     target_img_path = image_path
#     # 添加一个图像预处理
#     target_img = deal_org_image(target_img_path)
#
#     search_dir = image_data_path
#     # 多线程处理匹配
#     results = []
#     # 创建一个线程池执行器来管理多个线程
#     # 这里设置 max_workers 参数为你想要的线程数
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         # 遍历目录下的所有图像文件
#         image_paths = [os.path.join(search_dir, filename) for filename in os.listdir(search_dir)
#                        if filename.endswith('.png') or filename.endswith('.jpg')]
#
#         # 提交每张图像的匹配任务到线程池
#         futures = [executor.submit(match_image, target_img, image_path) for image_path in image_paths]
#
#         # 获取所有任务的结果
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             results.append(result)
#
#     # 找出最佳匹配图像路径
#     best_match_result = max(results, key=lambda x: x[1])
#
#     best_match_path = best_match_result[0]
#     best_inliers = best_match_result[1]
#     best_kp1 = best_match_result[2]
#     best_kp2 = best_match_result[3]
#     best_good_matches = best_match_result[4]
#     best_matches_mask = best_match_result[5]
#
#     # 读取最佳匹配图像
#     best_img = cv2.imread(best_match_path, cv2.IMREAD_GRAYSCALE)
#
#     # 绘制最佳匹配图像的特征点匹配
#     # if best_matches_mask is not None:
#     #     img_matches = cv2.drawMatches(target_img, best_kp1, best_img, best_kp2, best_good_matches, None,
#     #                                   matchesMask=best_matches_mask,
#     #                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     #     # 设置窗口标题为中文
#     #     cv2.imshow('最佳特征点匹配', img_matches)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
#
#     # 打印最佳匹配图像路径和匹配点数
#     # print(f"最佳匹配效果路径: {best_match_path}")
#     # print(f"匹配点数: {best_inliers}")
#     return best_match_path


import cv2
import os
import concurrent.futures
import numpy as np

def find_best_match(target_img_path, search_dir, max_workers=4):
    # 读取目标图像（灰度图像）
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.bilateralFilter(target_img, 9, 55, 55)

    # 调整图像尺寸的函数
    def resize(img, target_size=(300, 300)):
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

    # 创建SIFT特征检测器
    sift = cv2.SIFT_create(nfeatures=100, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=40, sigma=1.6)

    # 创建暴力匹配器
    bf = cv2.BFMatcher()

    # 定义匹配函数
    def match_image(image_path):
        try:
            # 读取待匹配图像（灰度图像）
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 调整图像尺寸
            img = resize(img, target_size=(300, 300))

            # 检测和计算SIFT特征
            kp1, des1 = sift.detectAndCompute(target_img, None)
            kp2, des2 = sift.detectAndCompute(img, None)

            # 使用暴力匹配器进行特征点匹配
            matches = bf.knnMatch(des1, des2, k=2)

            # 应用比例测试来筛选好的匹配点
            good_matches = []
            for match in matches:
                if len(match) == 2:  # 确保每个匹配元素都是一个长度为2的列表
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # 确保匹配点对的数量足够用于RANSAC
            if len(good_matches) >= 4:  # 至少需要4对匹配点
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 使用RANSAC算法进行匹配点筛选
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # 计算有效匹配点数
                num_inliers = np.sum(matches_mask)
            else:
                matches_mask = None
                num_inliers = 0

            return image_path, num_inliers, kp1, kp2, good_matches, matches_mask

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return image_path, 0, [], [], [], []

    # 多线程处理匹配
    results = []
    # 创建一个线程池执行器来管理多个线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 遍历目录下的所有图像文件
        image_paths = [os.path.join(search_dir, filename) for filename in os.listdir(search_dir)
                       if filename.endswith('.png') or filename.endswith('.jpg')]

        # 提交每张图像的匹配任务到线程池
        futures = [executor.submit(match_image, image_path) for image_path in image_paths]

        # 获取所有任务的结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)

    # 找出最佳匹配图像路径
    best_match_result = max(results, key=lambda x: x[1])

    best_match_path = best_match_result[0]
    best_inliers = best_match_result[1]
    best_kp1 = best_match_result[2]
    best_kp2 = best_match_result[3]
    best_good_matches = best_match_result[4]
    best_matches_mask = best_match_result[5]

    # 读取最佳匹配图像
    best_img = cv2.imread(best_match_path, cv2.IMREAD_GRAYSCALE)

    # 绘制最佳匹配图像的特征点匹配
    if best_matches_mask is not None:
        img_matches = cv2.drawMatches(target_img, best_kp1, best_img, best_kp2, best_good_matches, None,
                                      matchesMask=best_matches_mask,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('best match', img_matches)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # 返回最佳匹配结果
    return best_match_path


# target_img_path = r'G:\New_workstations\Course\zongsan\Detected_images\exp_test7\crops\warning\01428_0.jpg'
# search_dir = r'G:\New_workstations\Course\zongsan\300x300_images_data\number_inf\JL_300x300'
#
# best_match_path, best_inliers = find_best_match(target_img_path, search_dir)
# print(f"最佳匹配效果路径: {best_match_path}")
# print(f"匹配点数: {best_inliers}")
