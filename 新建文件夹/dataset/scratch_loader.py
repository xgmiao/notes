import os
import cv2
import torch
import collections
import scipy.misc as misc

from PIL import ImageEnhance, ImageFilter
from torch.utils import data
from dataset.augmentations import *


class ScratchLoader(data.Dataset):
    def __init__(self, root, split="train", num_class=2, img_size=(448, 448),
                 mean=np.array([100.5268, 100.5268, 100.5268]),
                 var=np.array([40.2457, 40.2457, 40.2457]),
                 img_norm=True, is_transform=False, augmentations=None, aug_ext=None):
        """
        ScratchLoader
        :param root:           (str)  root of the data set
        :param split:          (str)  dataset split, can be 'train' 'trainval' or 'val'
        :param num_class:      (int)  number of the class included in the dataset
        :param img_size:       (tuple or int) the size of the image for the deep model input
        :param img_norm:       (bool) norm the input image or not
        :param is_transform:   (bool) transform the input image or not
        :param augmentations:  (object) data augmentations used in the image and label
        """
        self.root = root  # root of data-set
        self.split = split  # 'train'or 'val'

        self.img_norm = img_norm
        self.num_class = num_class
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.is_transform = is_transform
        self.augmentations = augmentations
        self.aug_ext = aug_ext

        self.mean = mean
        self.var = var

        self.files = collections.defaultdict(list)

        for split in ['train', 'val']:
            path = os.path.join("/home/liuhuijun/PycharmProjects/S3Net/dataset", split + '.txt')  # path of split txt file

            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]

            self.files[split] = file_list

    def __len__(self):
        """
        Mount of the training or val files
        :return:
        """
        return len(self.files[self.split])

    def __getitem__(self, index):
        """
        Get one item from datasets during training or validation
        :param index:
        :return:
        """
        img_name = self.files[self.split][index]
        msk_name = img_name.replace(".bmp", ".png")

        image_path = os.path.join(self.root, self.split, img_name)
        label_path = os.path.join(self.root, self.split, msk_name)

        assert os.path.exists(os.path.join(label_path)), \
            "> Corresponding Mask: {} do not exist!!!".format(msk_name)

        image = misc.imread(image_path)
        image = np.array(image, dtype=np.uint8)

        # image = Image.fromarray(image, mode='RGB')

        # bright_enhancer = ImageEnhance.Brightness(image)
        # image = bright_enhancer.enhance(1.25)
        #
        # con_enhancer = ImageEnhance.Contrast(image)
        # image = con_enhancer.enhance(1.75)

        # sharp_enhancer = ImageEnhance.Sharpness(image)
        # image = sharp_enhancer.enhance(2.25)

        # image = image.filter(ImageFilter.EMBOSS)

        # image = np.array(image, dtype=np.uint8)
        image = image[:, :, ::-1]  # From RGB to BGR

        # Histogram Equalization
        # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        # image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
        # image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])

        label = misc.imread(label_path, mode="L")
        label[label > 0] = 1
        label = np.array(label, dtype=np.uint8)

        # data augmentation used in training
        if self.aug_ext is not None:
            image = self.aug_ext(image)
        if self.augmentations is not None:
            image, label = self.augmentations(image, label)

        if self.is_transform:
            image = self.transform(image)

        image = image.transpose(2, 0, 1)  # From HWC to CHW (For PyTorch we use N*C*H*W tensor)
        return torch.from_numpy(image).float(), torch.from_numpy(label).long()

    def transform(self, image):
        image = image.astype(float)
        image -= self.mean

        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            image = image.astype(float) / 255.0
            # image = image.astype(float) / self.var
        return image


if __name__ == "__main__":
    import cv2

    net_h, net_w = 448, 448
    augment = Compose([RandomHorizontallyFlip(),  RandomRotate(90), RandomSized((1.0, 1.35)),
                       RandomCrop((net_h, net_w))])

    root_dir = "/home/liuhuijun/PycharmProjects/S3Net/dataset/scratch"
    dataset = ScratchLoader(root=root_dir, split="val", num_class=2, img_size=(net_h, net_w),
                            img_norm=True, is_transform=False, augmentations=augment, aug_ext=None)

    bs = 1
    train_loader = data.DataLoader(dataset=dataset, batch_size=bs, num_workers=1, shuffle=True)
    for i, data in enumerate(train_loader):
        print("batch :", i)

        images, labels = data
        images = images.numpy()
        images = images.transpose(0, 2, 3, 1)[:, :, :, ::-1]  # From PyTorch Tensor to Numpy NArray, From BGR to RGB
        images = np.squeeze(images.astype(np.uint8))

        img_copy = images.copy()

        labels = labels.numpy() * 255  # From PyTorch Tensor to Numpy NArray
        labels = np.squeeze(labels.astype(np.uint8))

        mask_color = np.zeros((net_h, net_w, 3), dtype=np.uint8)
        colors = (0, 0, 128)

        if np.count_nonzero(labels) is not 0:
            mask_color[labels == 255] = colors
            img_copy[labels != 0] = cv2.addWeighted(img_copy[labels != 0].astype(np.uint8), 0.5,
                                                    mask_color[labels != 0].astype(np.uint8), 0.5, 0)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", images)

        cv2.namedWindow("ImageMask", cv2.WINDOW_NORMAL)
        cv2.imshow("ImageMask", img_copy)

        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", labels)
        cv2.waitKey(0)
