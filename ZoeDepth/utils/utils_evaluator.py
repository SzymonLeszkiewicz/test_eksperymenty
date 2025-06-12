from collections import defaultdict
import torch
import numpy as np

class DepthEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)

    def compute_metrics(self, pred, gt, valid_mask=None):
        if valid_mask is None:
            valid_mask = gt > 0

        pred = pred[valid_mask]
        gt = gt[valid_mask]

        rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
        l1 = torch.mean(torch.abs(gt - pred))
        thresh = torch.max((gt / pred), (pred / gt))
        delta1 = (thresh < 1.25).float().mean()
        delta2 = (thresh < 1.25 ** 2).float().mean()
        delta3 = (thresh < 1.25 ** 3).float().mean()

        return {
            'RMSE': rmse.item(),
            'MAE': l1.item(),
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item()
        }

    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.metrics[k].append(v)

    def get_results(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}