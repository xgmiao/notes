import numpy as np


def calc_median_frequency(classes, present_num):
    """
    Class balancing by median frequency balancing method.
    Reference: https://arxiv.org/pdf/1411.4734.pdf
       'a = median_freq / freq(c) where freq(c) is the number of pixels
        of class c divided by the total number of pixels in images where
        c is present, and median_freq is the median of these frequencies.'
    """
    class_freq = classes / present_num
    median_freq = np.median(class_freq)
    return median_freq / class_freq


def calc_log_frequency(classes, value=1.02):
    """Class balancing by ERFNet method.
       prob = each_sum_pixel / each_sum_pixel.max()
       a = 1 / (log(1.02 + prob)).
    """
    class_freq = classes / classes.sum()  # ERFNet is max, but ERFNet is sum
    # print(class_freq)
    # print(np.log(value + class_freq))
    return 1 / np.log(value + class_freq)


if __name__ == '__main__':
    import os
    from torch.utils import data
    from dataset.scratch_loader import ScratchLoader
    from dataset.augmentations import *

    method = "median"
    result_path = "/home/liuhuijun/PycharmProjects/S3Net/dataset"

    num_classes = 2
    net_h, net_w = 448, 448

    local_path = "/home/liuhuijun/PycharmProjects/S3Net/dataset/scratch"

    augment = Compose([RandomCrop((net_h, net_w))])
    dataset = ScratchLoader(root=local_path, split="train", num_class=2, img_size=(net_h, net_w),
                            mean=np.array([68.4067,  68.4184,  68.4502]),
                            img_norm=False, is_transform=False, augmentations=augment, aug_ext=None)

    classes, present_num = ([0 for i in range(num_classes)] for i in range(2))

    bs = 1
    train_loader = data.DataLoader(dataset=dataset, batch_size=bs, num_workers=1, shuffle=False)
    for i, data in enumerate(train_loader):
        print("batch :", i)

        images, labels = data

        labels = labels.numpy()  # From PyTorch Tensor to Numpy NArray

        for nc in range(num_classes):
            num_pixel = (labels == nc).sum()
            if num_pixel:
                classes[nc] += num_pixel
                present_num[nc] += 1

    if 0 in classes:
        raise Exception("Some classes are not found")

    classes = np.array(classes, dtype="f")
    presetn_num = np.array(classes, dtype="f")
    if method == "median":
        class_weight = calc_median_frequency(classes, present_num)
    elif method == "log":
        class_weight = calc_log_frequency(classes)
    else:
        raise Exception("Please assign method to 'mean' or 'log'")

    print("class weight", class_weight)
    result_path = os.path.join(result_path, "{}_class_weight.npy".format(method))
    np.save(result_path, class_weight)
    print("Done!")
