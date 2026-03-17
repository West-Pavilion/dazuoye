# coding:utf-8
# PROJECT_NAME: rice_growth_stages_classification
# Author: Zhang Qiangzhi
# DATE: 2023-03-16
# -------------------------------------
import os
# import platform
import shutil
from PIL import Image
from torchvision import transforms as tfs


# src_dir = f"../dataset_before"    # 原始数据所在文件夹:设置相对路径或者绝对路径
# dst_dir = f"../dataset_augmented"  # 增强后存放目的地：设置相对路径或者绝对路径
def data_augment(src_dir):
    # # 不同操作系统不同路径分隔符
    # SYSTEM_PLATFORM = platform.system()
    # if SYSTEM_PLATFORM == "Windows":
    #     split_str = "\\"
    # else:
    #     split_str = "/"

    # 遍历文件
    dir_file = os.walk(src_dir)
    for path, dir_list, file_list in dir_file:
        for file_name in file_list:
            last_dir = os.path.basename(path)
            path_new = f"{src_dir}_augmented/{last_dir}"

            if not os.path.exists(path_new):
                os.makedirs(path_new)

            file_name_full = os.path.join(path, file_name)
            img = Image.open(file_name_full)

            img_color = tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)
            # img_bright = tfs.ColorJitter(brightness=0.5)(img)
            # img_contrast = tfs.ColorJitter(contrast=0.5)(img)
            # img_saturation = tfs.ColorJitter(saturation=0.5)(img)
            # img_hue = tfs.ColorJitter(hue=0.5)(img)

            # img_vertical = tfs.RandomVerticalFlip(p=1)(img)
            # img_horizontal = tfs.RandomHorizontalFlip(p=1)(img)

            # img_rotation = tfs.RandomRotation(45)(img)

            img_affine = tfs.RandomAffine(degrees=35)(img)
            # img_translate = tfs.RandomAffine(degrees=0, translate=(0.3, 0.3))(img)
            # img_scale = tfs.RandomAffine(degrees=0, scale=(0.7, 0.7))(img)

            file_name_new = f"{path_new}\\{file_name}"
            shutil.copy(file_name_full, file_name_new)

            img_color.save(f"{file_name_new}_color.jpg")
            # img_vertical.save(f"{file_name_new}_vertical.jpg")
            # img_horizontal.save(f"{file_name_new}_horizontal.jpg")
            img_affine.save(f"{file_name_new}_affine.jpg")
            # img_rotation.save(f"{file_name_new}_rotation.jpg")
            # img_translate.save(f"{file_name_new}_translate.jpg")
            # img_scale.save(f"{file_name_new}_scale.jpg")

            if last_dir == "Bacterial_leaf_blight":
                img_perspective = tfs.RandomPerspective(distortion_scale=0.2, p=1)(img)
                img_perspective.save(f"{file_name_new}_perspective.jpg")

    return f"{src_dir}_augmented"


if __name__ == '__main__':
    result = data_augment(f"F:/Diseases_and_Pests_20230903_V2/dataset_classification")
    print(result)
    # result1 = data_augment(f"E:/Diseases_and_Pests_20230903_V2/train")
    # print(result1)
    # result2 = data_augment(f"E:/Diseases_and_Pests_20230903_V2/val")
    # print(result2)
