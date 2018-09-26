import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Bootstrapped Cross Entropy 2D
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def bootstrapped_binary_cross_entropy2d(input, target, K, weight=None):
    """
            A categorical cross entropy loss for 4D tensors.
            We assume the following layout: (batch, classes, height, width)
            Args:
                input: The outputs.
                target: The predictions.
                K: The number of pixels to select in the bootstrapping process.
                   The total number of pixels is determined as 512 * multiplier.
            Returns:
                The pixel-bootstrapped binary cross entropy loss.
        """
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None):
        input = input.view(1, -1)
        target = target.view(1, -1)

        # torch.ones_like(input=target, dtype=torch.float32, requires_grad=True)
        weights = torch.index_select(input=weight, dim=0, index=torch.squeeze(target))

        loss = F.binary_cross_entropy(input=input, target=target.float(), weight=weights,
                                      size_average=False, reduce=False)
        topk_loss, _ = loss.topk(K)
        return topk_loss.sum() / K

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight)
    return loss / float(batch_size)


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=False):
    """
        A categorical cross entropy loss for 4D tensors.
        We assume the following layout: (batch, classes, height, width)
        Args:
            input: The outputs.
            target: The predictions.
            K: The number of pixels to select in the bootstrapping process.
               The total number of pixels is determined as 512 * multiplier.
        Returns:
            The pixel-bootstrapped cross entropy loss.
    """
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=False):
        n, c, h, w = input.size()

        # 1. The log softmax. log_p: (n, c, h, w)
        # input = F.normalize(input, p=2, dim=1, eps=1e-5)
        log_p = F.log_softmax(input, dim=1)

        # 2. log_p: (n*h*w, c) - contiguous() required if transpose() is used before view().
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        # 3. target: (n*h*w,)
        mask = target >= 0
        target = target[mask]

        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250, reduction="none")

        # For each element in the batch, collect the top K worst predictions
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Dice Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def binary_dice_loss(inputs, targets, weight=None):
    """
        inputs : NxCxHxW Variable
        targets :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
    smooth = 1.0
    # inputs = F.logsigmoid(inputs).exp()

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    weight = torch.index_select(input=weight, dim=0, index=targets)

    targets = targets.float()
    numerator = 2.0 * (weight * inputs * targets).sum() + smooth
    denominator = (weight * inputs).sum() + (weight * targets).sum() + smooth

    return 1.0 - (numerator / denominator)


def dice_loss(inputs, targets, weights=None, ignore_index=None):
    """
    inputs : NxCxHxW Variable
    targets :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1.0
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    numerator = 2.0 * intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    loss_per_channel = weights * (1.0 - (numerator / denominator))

    return loss_per_channel.sum() / inputs.size(1)


def bootstrapped_dice_loss(inputs, targets, weights=None, top_k=128, ignore_index=None):
    """
        inputs : NxCxHxW Variable
        targets :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
    smooth = 1.0
    num_cls = inputs.size(1)
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.softmax(inputs, dim=1)
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    numerator = intersection.sum(dim=0)
    denominator = denominator.sum(dim=0)
    numerator = numerator.transpose(0, 1).transpose(1, 2).contiguous().view(-1, num_cls)
    denominator = denominator.transpose(0, 1).transpose(1, 2).contiguous().view(-1, num_cls)

    loss_per_channel = weights * (1.0 - ((2.0 * numerator + smooth) / (denominator + smooth)))
    loss = loss_per_channel.sum(dim=1) / num_cls

    topk_loss, _ = loss.topk(top_k)
    topk_loss = topk_loss.sum() / top_k

    return topk_loss


def soft_jaccard_loss(inputs, targets, weights=None, ignore_index=None):
    """
    inputs : NxCxHxW Variable
    targets :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1.0
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1)
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    denominator = denominator.sum(dim=0).sum(dim=1).sum(dim=1) + smooth
    # loss_per_channel = weights * (1.0 - torch.log(((numerator + smooth) / (denominator - numerator + smooth))))
    loss_per_channel = weights * (1.0 - ((numerator + smooth) / (denominator - numerator + smooth)))

    return loss_per_channel.sum() / inputs.size(1)


def bootstrapped_jaccard_loss(inputs, targets, weights=None, top_k=128, ignore_index=None):
    """
    inputs : NxCxHxW Variable
    targets :  NxHxW LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    """
    smooth = 1.0
    num_cls = inputs.size(1)
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.softmax(inputs, dim=1)
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = inputs * encoded_target
    denominator = inputs + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0

    numerator = intersection.sum(dim=0)
    denominator = denominator.sum(dim=0)
    numerator = numerator.transpose(0, 1).transpose(1, 2).contiguous().view(-1, num_cls)
    denominator = denominator.transpose(0, 1).transpose(1, 2).contiguous().view(-1, num_cls)

    loss_per_channel = weights * (1.0 - ((numerator + smooth) / (denominator - numerator + smooth)))
    loss = loss_per_channel.sum(dim=1) / num_cls

    topk_loss, _ = loss.topk(top_k)
    topk_loss = topk_loss.sum() / top_k

    return topk_loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Tversky Loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def tversky_loss(inputs, targets, alpha=0.4, beta=0.6, weights=None, ignore_index=None):
    """
        inputs : NxCxHxW Variable
        targets :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
    smooth = 1.0
    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    ones = torch.ones_like(inputs)

    intersection = inputs * encoded_target
    numerator = intersection.sum(dim=0).sum(dim=1).sum(dim=1) + smooth

    item1 = inputs * (ones - encoded_target)
    item2 = (ones - inputs) * encoded_target
    denominator = numerator + alpha * item1.sum(dim=0).sum(dim=1).sum(dim=1) + \
                  beta * item2.sum(dim=0).sum(dim=1).sum(dim=1)

    if ignore_index is not None:
        denominator[mask] = 0

    num_cls = inputs.size(1)
    tversky_index = weights * (1 - (numerator / denominator))

    return tversky_index.sum() / num_cls


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Asymmetric Similarity Loss Function to Balance Precision and Recall in
# Highly Unbalanced Deep Medical Image Segmentation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def asymmetric_similarity_loss(inputs, targets, beta=1.725, weights=None, ignore_index=None):
    """
        inputs : NxCxHxW Variable
        targets :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
    eps = 1e-8
    beta = beta * beta

    # inputs = F.normalize(inputs, p=2, dim=1, eps=1e-5)
    inputs = F.log_softmax(inputs, dim=1).exp()
    encoded_target = inputs.detach() * 0  # The result will never require gradient.

    if ignore_index is not None:
        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0

        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)

    if weights is None:
        weights = 1.0

    ones = torch.ones_like(inputs).float()

    intersection = inputs * encoded_target
    numerator = (1.0 + beta) * intersection.sum(dim=0).sum(dim=1).sum(dim=1)

    item1 = (ones - inputs) * encoded_target
    item2 = inputs * (ones - encoded_target)
    denominator = numerator + beta * item1.sum(dim=0).sum(dim=1).sum(dim=1) + \
                  item2.sum(dim=0).sum(dim=1).sum(dim=1) + eps

    if ignore_index is not None:
        denominator[mask] = 0

    num_cls = inputs.size(1)
    f_beta_loss = weights * (1.0 - (numerator / denominator))

    return f_beta_loss.sum()/num_cls


class FocalLoss2D(nn.Module):
    """
    Focal Loss, which is proposed in:
        "Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002v2)"
    """
    def __init__(self, num_classes=2, weights=None, ignore_label=250,
                 alpha=0.25, gamma=2, size_average=True):
        """
        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        :param num_classes:   (int) num of the classes
        :param ignore_label:  (int) ignore label
        :param alpha:         (1D Tensor or Variable) the scalar factor
        :param gamma:         (float) gamma > 0;
                                      reduces the relative loss for well-classified examples (probabilities > .5),
                                      putting more focus on hard, mis-classified examples
        :param size_average:  (bool): By default, the losses are averaged over observations for each mini-batch.
                                      If the size_average is set to False, the losses are
                                      instead summed for each mini-batch.
        """
        super(FocalLoss2D, self).__init__()
        self.weights = weights

        self.alpha = alpha
        self.gamma = gamma

        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.size_average = size_average
        self.one_hot = torch.eye(self.num_classes)

    def _bootstrap_focal_loss(self, cls_preds, cls_targets, top_k=128):
        n, c, h, w = cls_preds.size()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. target reshape and one-hot encode
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1.1. target: (n*h*w,)
        cls_targets = cls_targets.view(n * h * w, 1)
        target_mask = (cls_targets >= 0) * (cls_targets != self.ignore_label)

        cls_targets = cls_targets[target_mask]
        cls_targets = self.one_hot.index_select(dim=0, index=cls_targets)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. compute focal loss for multi-classification
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2.1. The softmax. prob: (n, c, h, w)
        prob = F.softmax(cls_preds, dim=1)
        # 2.2. prob: (n*h*w, c) - contiguous() required if transpose() is used before view().
        prob = prob.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        prob = prob[target_mask.repeat(1, c)]
        prob = prob.view(-1, c)  # (n*h*w, c)

        probs = torch.clamp((self.weights * prob * cls_targets).sum(1).view(-1), min=1e-8, max=1.0)
        loss = -self.alpha * (torch.pow((1 - probs), self.gamma)) * probs.log()

        # For each element in the batch, collect the top K worst predictions
        topk_loss, _ = loss.topk(top_k)
        reduced_topk_loss = topk_loss.sum() / top_k

        return reduced_topk_loss

    def forward(self, cls_preds, cls_targets, K=128, weight=None):
        """

        :param cls_preds:    (n, c, h, w)
        :param cls_targets:  (n, h, w)
        :return:
        """
        assert not cls_targets.requires_grad
        assert cls_targets.dim() == 3
        assert cls_preds.size(0) == cls_targets.size(0), "{0} vs {1} ".format(cls_preds.size(0), cls_targets.size(0))
        assert cls_preds.size(2) == cls_targets.size(1), "{0} vs {1} ".format(cls_preds.size(2), cls_targets.size(1))
        assert cls_preds.size(3) == cls_targets.size(2), "{0} vs {1} ".format(cls_preds.size(3), cls_targets.size(3))

        if cls_preds.is_cuda:
            self.one_hot = self.one_hot.cuda()

        batch_size = cls_preds.size(0)

        final_loss = 0.0
        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            final_loss += self._bootstrap_focal_loss(torch.unsqueeze(cls_preds[i], 0),
                                                     torch.unsqueeze(cls_targets[i], 0),
                                                     top_k=K)

        if self.size_average:
            final_loss /= batch_size

        return final_loss


if __name__ == "__main__":
    while True:
        dummy_in = torch.randn(32, 2, 32, 32).requires_grad_()
        dummy_gt = torch.LongTensor(32, 32, 32).random_(0, 2)

        top_k = 256

        class_weight = np.array([0.50374176, 67.31353153], dtype=float)  # 0.50430964, 58.50954048
        class_weight = torch.from_numpy(class_weight).float()

        """
        loss = F.binary_cross_entropy_with_logits(input=input, target=target.float(), weight=weights,
                                                  size_average=True, reduce=False)
        topk_loss, _ = loss.topk(top_k)
        loss = topk_loss.sum() / top_k
        """

        # loss = bootstrapped_binary_cross_entropy2d(F.sigmoid(dummy_in), dummy_gt, K=top_k, weight=class_weight)
        loss = bootstrapped_dice_loss(dummy_in, dummy_gt, weights=class_weight)

        print("Loss: {}".format(loss.item()))
