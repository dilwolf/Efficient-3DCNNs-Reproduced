"""
Utility classes and functions for video classification training.
"""
import os
import csv
import torch
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class Logger:
    """
    Lightweight TSV logger for epoch/batch metrics.

    Usage:
        with Logger(path, ['epoch', 'loss', 'prec1']) as log:
            log.log({'epoch': 1, 'loss': 0.5, 'prec1': 72.3})
    """

    def __init__(self, path: str, header: list[str]):
        self.path   = Path(path)
        self.header = header
        self._file  = open(self.path, 'w', newline='')
        self._writer = csv.writer(self._file, delimiter='\t')
        self._writer.writerow(header)
        self._file.flush()

    def log(self, values: dict):
        for col in self.header:
            assert col in values, f"Missing key '{col}' in log values"
        self._writer.writerow([values[col] for col in self.header])
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)):
    """
    Computes precision@k for the specified values of k.

    Args:
        output: Model logits of shape (N, C)
        target: Ground truth labels of shape (N,)
        topk:   Tuple of k values to evaluate

    Returns:
        List of scalar accuracy values (one per k), each in range [0, 100]
    """
    with torch.no_grad():
        maxk       = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()                                   # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res


def save_checkpoint(state: dict, is_best: bool, opt):
    """
    Save training checkpoint. If is_best, also copy to best model file.

    Args:
        state:   Dictionary containing model/optimizer/scheduler state
        is_best: Whether this is the best model so far
        opt:     Options object with result_path and store_name attributes
    """
    result_path = Path(opt.result_path)
    checkpoint_path = result_path / f"{opt.store_name}_last.pth"
    best_path       = result_path / f"{opt.store_name}_best.pth"

    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        logger.info(f"Best model updated: {best_path}")