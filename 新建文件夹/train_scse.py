import torch.nn.functional as F
import argparse
import torch
import time
import os

from dataset.scratch_loader import ScratchLoader
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

from utils.loss import dice_loss, soft_jaccard_loss, bootstrapped_cross_entropy2d, FocalLoss2D
from utils.utils import cosine_annealing_lr, set_optimizer_lr
from utils.metrics import RunningScore
from dataset.augmentations import *
from models.scseliunetv2 import SCSELiuNetV2


def train(args, data_root, save_root):
    weight_dir = "{}weights/".format(save_root)
    log_dir = "{}logs/ScratchLiuNet-{}".format(save_root, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup Augmentations
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    net_h, net_w = args.img_rows, args.img_cols

    augment_train = Compose([RandomHorizontallyFlip(), RandomRotate(90), RandomCrop((net_h, net_w))])
    augment_valid = Compose([CenterCrop((net_h, net_w))])

    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 0. Setting up DataLoader...")
    train_loader = ScratchLoader(root=data_root, split="train", num_class=2, img_size=(net_h, net_w), img_norm=True,
                                 is_transform=True, augmentations=augment_train)
    valid_loader = ScratchLoader(root=data_root, split="val", num_class=2, img_size=(net_h, net_w), img_norm=True,
                                 is_transform=True, augmentations=augment_valid)

    n_classes = train_loader.num_class

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Setup Metrics
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    running_metrics = RunningScore(n_classes)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    model = SCSELiuNetV2(num_classes=n_classes, in_size=(net_h, net_w), width_mult=2.0)

    # np.arange(torch.cuda.device_count())
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # 4.1 Setup Optimizer
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.90,
                                    weight_decay=5e-4, nesterov=True)

        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
        #                             eps=1e-08, weight_decay=0, amsgrad=True)
        # optimizer = YFOptimizer(model.parameters(), lr=2.5e-3, mu=0.9, clip_thresh=10000, weight_decay=5e-4)

    # 4.2 Setup Loss
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    class_weight = None
    ce_loss = None
    jac_loss = None
    if hasattr(model.module, 'loss'):
        print('> Using custom loss')
        loss_fn = model.module.loss
    else:
        class_weight = np.array([0.50603186, 41.94659861], dtype=float)  # 0.50928796, 27.41656205
        class_weight = torch.from_numpy(class_weight).float().cuda()
        loss_fn = dice_loss
        jac_loss = soft_jaccard_loss
        ce_loss = bootstrapped_cross_entropy2d
        # ce_loss = FocalLoss2D(num_classes=2, weights=class_weight, ignore_label=250,
        #                       alpha=0.25, gamma=2, size_average=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 5. Resume Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    best_iou = -100.0
    cor_precision = -100.0
    cor_recall = -100.0
    args.start_epoch = 0
    if args.resume is not None:
        full_path = "{}{}".format(weight_dir, args.resume)
        if os.path.isfile(full_path):
            print("> Loading model and optimizer from checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(full_path)
            args.start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['model_state'])          # weights
            optimizer.load_state_dict(checkpoint['optimizer_state'])  # gradient state

            del checkpoint
            print("> Loaded checkpoint '{}' (epoch {}, iou {})".format(args.resume,
                                                                       args.start_epoch,
                                                                       best_iou))

        else:
            print("> No checkpoint found at '{}'".format(args.resume))
    else:
        if args.pre_trained is not None:
            print("> Loading weights from pre-trained model '{}'".format(args.pre_trained))
            full_path = "{}{}".format(weight_dir, args.pre_trained)

            pre_weight = torch.load(full_path)
            pre_weight = pre_weight["model_state"]

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pre_weight.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            del pre_weight
            del pretrained_dict

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 3. Setup tensor_board for visualization
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    writer = None
    if args.tensor_board:
        writer = SummaryWriter(log_dir=log_dir, comment="ScratchLiuNet")

        # dummy_input = torch.rand(1, 3, net_h, net_w).cuda()
        # writer.add_graph(model, dummy_input)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 6. Train Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 2. Model Training start...")
    train_loader = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_loader, batch_size=args.batch_size, num_workers=8, shuffle=False)

    num_batches = int(math.ceil(len(train_loader.dataset.files[train_loader.dataset.split]) /
                                float(train_loader.batch_size)))

    # lr_period = 20 * num_batches

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    # scheduler = CyclicLR(optimizer, base_lr=1.0e-3, max_lr=6.0e-3, step_size=2*num_batches)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.25)

    alpha = 0.7
    beta = 0.96
    topk_base = 196
    for epoch in np.arange(args.start_epoch, args.n_epoch):
        scheduler.step()
        # scheduler.batch_step()

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 7.1 Mini-Batch Learning
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        model.train()

        last_loss = 0.0
        pbar = tqdm(np.arange(num_batches))
        for train_i, (images, labels) in enumerate(train_loader):  # One mini-Batch data, One iteration
            full_iter = (epoch * num_batches) + train_i + 1

            # batch_lr = args.l_rate * cosine_annealing_lr(lr_period, full_iter)
            # optimizer = set_optimizer_lr(optimizer, batch_lr)

            images = images.cuda().requires_grad_()   # Image feed into the deep neural network
            labels = labels.cuda()

            optimizer.zero_grad()
            net_out = model(images)

            train_loss = loss_fn(net_out, labels, weights=class_weight) + \
                         (1.0 - alpha) * ce_loss(net_out, labels, K=topk_base*topk_base, weight=class_weight)
            loss = train_loss.item()
            last_loss += loss
            train_loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, args.n_epoch))
            pbar.set_postfix(Loss=loss)

            if (train_i + 1) % 30 == 0 and args.tensor_board:
                loss_log = "Epoch [%d/%d], Iter: %d Loss: \t %.4f" % (epoch + 1, args.n_epoch,
                                                                      train_i + 1, last_loss / (train_i + 1))

                pred = F.softmax(net_out, dim=1).argmax(dim=1).cpu().numpy()
                gt = labels.cpu().numpy()

                running_metrics.update(gt, pred)
                score, class_iou, class_acc, class_prc, class_rcl, cls_f1 = running_metrics.get_scores()

                metric_log = ""
                for k, v in score.items():
                    metric_log += " {}: \t %.4f, ".format(k) % v
                running_metrics.reset()

                logs = loss_log + metric_log
                # print(logs)

                if args.tensor_board:
                    writer.add_scalar('Training/Losses', last_loss / (train_i + 1), full_iter)
                    writer.add_scalars('Training/Metrics', score, full_iter)
                    writer.add_scalars('Training/ClassIOU', class_iou, full_iter)
                    writer.add_scalars('Training/ClassAcc', class_acc, full_iter)
                    writer.add_text('Training/Text', logs, full_iter)

                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), full_iter)

        last_loss /= num_batches
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 7.2 Mini-Batch Validation
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # print("> Validation for Epoch [%d/%d]:" % (epoch + 1, args.n_epoch))
        model.eval()

        mval_loss = 0.0
        vali_count = 0

        with torch.no_grad():
            for i_val, (images, labels) in enumerate(valid_loader):
                vali_count += 1

                images = images.cuda()
                labels = labels.cuda()

                net_out = model(images)

                val_loss = loss_fn(net_out, labels, weights=class_weight) + \
                           (1.0 - alpha) * ce_loss(net_out, labels, K=topk_base*topk_base, weight=class_weight)
                mval_loss += val_loss.item()

                pred = F.softmax(net_out, dim=1).argmax(dim=1).cpu().numpy()
                gt = labels.cpu().numpy()
                running_metrics.update(gt, pred)

            mval_loss /= vali_count

            loss_log = "Epoch [%d/%d] Loss: \t %.4f" % (epoch + 1, args.n_epoch, mval_loss)
            metric_log = ""

            score, class_iou, class_acc, class_prc, class_rcl, cls_f1 = running_metrics.get_scores()
            for k, v in score.items():
                metric_log += " {} \t %.4f, ".format(k) % v
            running_metrics.reset()

            logs = loss_log + metric_log
            # print(logs)
            pbar.set_postfix(Train_Loss=last_loss, Vali_Loss=mval_loss,
                             IoU0=class_iou["Class#0"], Acc0=class_acc["Class#0"],
                             IoU1=class_iou["Class#1"], Acc1=class_acc["Class#1"],
                             Precision1=class_prc["Class#1"], Recall1=class_rcl["Class#1"])

            if args.tensor_board:
                writer.add_scalar('Validation/Losses', mval_loss, epoch)
                writer.add_scalars('Validation/Metrics', score, epoch)
                writer.add_scalars('Validation/ClassIOU', class_iou, epoch)
                writer.add_scalars('Validation/ClassAcc', class_acc, epoch)
                writer.add_text('Validation/Text', logs, epoch)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                # export scalar data to JSON for external processing
                # writer.export_scalars_to_json("{}/all_scalars.json".format(log_dir))

            if class_iou["Class#1"] >= best_iou:
                best_iou = class_iou["Class#1"]
                cor_precision = class_prc["Class#1"]
                cor_recall = class_rcl["Class#1"]

                state = {'epoch': epoch + 1,
                         "best_iou": best_iou,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, "{}{}_scseliunetv2_best_model.pkl".format(weight_dir, args.dataset))

        pbar.close()

    if args.tensor_board:
        # export scalar data to JSON for external processing
        # writer.export_scalars_to_json("{}/all_scalars.json".format(log_dir))
        writer.close()

    print("")
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Best_IoU: {}, Cor_Precision: {}, Cor_Recall: {}".format(best_iou, cor_precision, cor_recall))
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Training Done!!!")
    print("> # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


if __name__ == '__main__':
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 0. Hyper-params
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    parser = argparse.ArgumentParser(description='Hyperparams')

    parser.add_argument('--dataset', nargs='?', type=str, default='scratch',
                        help='Dataset to use')
    parser.add_argument('--img_rows', nargs='?', type=int, default=448,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=448,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=128,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=7.5e-3,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pre_trained', nargs='?', type=str, default="scratch_scseliunetv2_best_model.pkl",
                        help='Path to pre-trained  model to init from')
    parser.add_argument('--tensor_board', nargs='?', type=bool, default=False,
                        help='Show visualization(s) on tensor_board | True by  default')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Train the Deep Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,3,2,1,0"
    data_path = "/home/liuhuijun/PycharmProjects/S3Net/dataset/scratch"
    save_path = "/home/liuhuijun/TrainLog/"
    train_args = parser.parse_args()
    train(train_args, data_path, save_path)
