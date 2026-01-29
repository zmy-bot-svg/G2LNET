#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchmetrics


class sMAPE(torchmetrics.Metric):
    """Symmetric mean absolute percentage error (sMAPE)."""
    full_state_update: bool = False
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update sMAPE state."""
        smape = 2 * (preds - target).abs() / (preds.abs() + target.abs() + self.eps)
        valid_mask = ~torch.isnan(smape)
        smape_valid = smape[valid_mask]
        
        if smape_valid.numel() > 0:
            self.sum += smape_valid.sum()
            self.count += smape_valid.numel()
    
    def compute(self):
        """Compute final sMAPE value."""
        if self.count == 0:
            return torch.tensor(0.0)
        return self.sum / self.count


class MultiTaskMAE(torchmetrics.Metric):
    """Multi-task mean absolute error."""
    full_state_update: bool = False
    
    def __init__(self, num_tasks: int, task_names: list = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(num_tasks)]
        
        for i in range(num_tasks):
            self.add_state(f"sum_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"count_{i}", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update MAE state per task."""
        if preds.dim() == 1 and target.dim() == 1:
            mae = torch.abs(preds - target)
            valid_mask = ~torch.isnan(mae)
            if valid_mask.any():
                getattr(self, "sum_0") += mae[valid_mask].sum()
                getattr(self, "count_0") += valid_mask.sum()
        elif preds.dim() == 2 and target.dim() == 2:
            for i in range(min(self.num_tasks, preds.shape[1])):
                mae_i = torch.abs(preds[:, i] - target[:, i])
                valid_mask = ~torch.isnan(mae_i)
                if valid_mask.any():
                    getattr(self, f"sum_{i}") += mae_i[valid_mask].sum()
                    getattr(self, f"count_{i}") += valid_mask.sum()
    
    def compute(self):
        """Compute overall mean MAE."""
        total_sum = torch.tensor(0.0)
        total_count = torch.tensor(0)
        
        for i in range(self.num_tasks):
            task_sum = getattr(self, f"sum_{i}")
            task_count = getattr(self, f"count_{i}")
            total_sum += task_sum
            total_count += task_count
        
        if total_count == 0:
            return torch.tensor(0.0)
        return total_sum / total_count
    
    def compute_per_task(self):
        """Compute MAE per task."""
        results = {}
        for i in range(self.num_tasks):
            task_sum = getattr(self, f"sum_{i}")
            task_count = getattr(self, f"count_{i}")
            if task_count > 0:
                results[self.task_names[i]] = (task_sum / task_count).item()
            else:
                results[self.task_names[i]] = 0.0
        return results


def create_multitask_metrics(num_tasks: int, task_names: list = None, use_smape: bool = True):
    """Create metric dict for multi-task learning."""
    metrics_dict = {
        "mae": MultiTaskMAE(num_tasks, task_names),
    }
    
    if use_smape:
        metrics_dict["smape"] = sMAPE()
    else:
        metrics_dict["mape"] = torchmetrics.MeanAbsolutePercentageError(eps=1e-7)
    
    return metrics_dict
