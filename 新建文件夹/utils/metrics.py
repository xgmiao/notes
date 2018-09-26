import numpy as np


class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.class_id = []
        for cls_id in range(self.n_classes):
            self.class_id.append("Class#{}".format(cls_id))

    @staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        tp = np.diag(hist)            # True Positives + True Negatives
        sum_a0 = hist.sum(axis=0)
        sum_a1 = hist.sum(axis=1)     # Condition Positives and Condition Negatives

        acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

        cls_acc = tp / (sum_a1 + np.finfo(np.float32).eps)
        mean_acc = np.nanmean(cls_acc)
        cls_acc = dict(zip(self.class_id, cls_acc))

        cls_prc = tp / (sum_a0 + np.finfo(np.float32).eps)
        cls_rcl = tp / (sum_a1 + np.finfo(np.float32).eps)
        cls_f1 = (2 * cls_prc * cls_rcl) / (cls_prc + cls_rcl + np.finfo(np.float32).eps)

        precision = cls_prc.sum()
        recall = cls_rcl.sum()
        f1 = (2 * precision * recall) / (precision + recall + np.finfo(np.float32).eps)

        cls_prc = dict(zip(self.class_id, cls_prc))
        cls_rcl = dict(zip(self.class_id, cls_rcl))
        cls_f1 = dict(zip(self.class_id, cls_f1))

        iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)

        cls_iu = dict(zip(self.class_id, iu))
        mean_iu = np.nanmean(iu)  # Compute the arithmetic mean along the specified axis, ignoring NaNs.

        freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        return {'Overall_Acc': acc,
                'Mean_Acc': mean_acc,
                'FreqW_Acc': fwavacc,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Mean_IoU': mean_iu}, cls_iu, cls_acc, cls_prc, cls_rcl, cls_f1

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


if __name__ == "__main__":
    n_class = 2
    score = RunningScore(n_class)

    label_true = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    label_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])

    score.update(label_true, label_pred)
    print(score.confusion_matrix)

