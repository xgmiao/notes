import os
import numpy as np

from utils.utils import recursive_glob

if __name__ == "__main__":
    root_dir = "/home/liuhuijun/PycharmProjects/S3Net/dataset/scratch"

    img_msk_path = os.path.join(root_dir, "train")
    img_list = recursive_glob(rootdir=img_msk_path, suffix=".bmp")

    val_img_path = os.path.join(root_dir, "val")
    val_img_list = recursive_glob(rootdir=val_img_path, suffix=".bmp")

    train_count = len(img_list)
    val_count = len(val_img_list)

    save_root = "/home/liuhuijun/PycharmProjects/S3Net/dataset"
    train_save = os.path.join(save_root, "train.txt")
    val_save = os.path.join(save_root, "val.txt")

    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Creating {}".format(train_save))

    shuffled_list = np.arange(train_count)
    np.random.shuffle(shuffled_list)
    with open(train_save, "w") as f:
        for idx in np.arange(train_count):
            file_idx = shuffled_list[idx]

            image_name = img_list[file_idx]
            image_name = image_name.split("/")[-1]
            mask_name = image_name.replace(".bmp", ".png")

            assert os.path.exists(os.path.join(img_msk_path, mask_name)), \
                "> Corresponding Mask: {} do not exist!!!".format(mask_name)

            f.write(image_name + os.linesep)

    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Creating {}".format(val_save))

    shuffled_list = np.arange(val_count)
    np.random.shuffle(shuffled_list)
    with open(val_save, "w") as f:
        for idx in np.arange(val_count):
            file_idx = shuffled_list[idx]

            image_name = val_img_list[file_idx]
            image_name = image_name.split("/")[-1]
            mask_name = image_name.replace(".bmp", ".png")

            assert os.path.exists(os.path.join(val_img_path, mask_name)), \
                "> Corresponding Mask: {} do not exist!!!".format(mask_name)

            f.write(image_name + os.linesep)

    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Done !!!")
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
