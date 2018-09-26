import torch
from dataset.scratch_loader import ScratchLoader
from dataset.augmentations import *


def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('> Computing mean and std of images in the dataset..')

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


if __name__ == "__main__":
    import time
    net_h, net_w = 448, 448

    local_path = "/home/liuhuijun/PycharmProjects/S3Net/dataset/scratch"
    augment_train = Compose([RandomHorizontallyFlip(), RandomRotate(90), RandomCrop((net_h, net_w))])
    dataset = ScratchLoader(root=local_path, split="train", num_class=2, img_size=(net_h, net_w), img_norm=False,
                            is_transform=False, augmentations=augment_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    count = 20
    mmean = torch.zeros(3)
    mstd = torch.zeros(3)
    time_cost = 0.0
    for idx in np.arange(count):
        print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
        print("> Epoch: {}...".format(idx + 1))
        start_time = time.time()
        mean, std = get_mean_and_std(dataloader)
        mmean = mmean + mean
        mstd = mstd + std

        end_time = time.time() - start_time
        time_cost += end_time
        print("> Time: {}..., ".format(end_time))
        print("> Mean: {}".format(mean))
        print("> Mean: {}".format(std))

    print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")
    print("> Done, Time: {}s".format(time_cost))
    print("> Mean: {}".format(mmean / count))
    print("> Mean: {}".format(mstd / count))
    print("> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++ <")


