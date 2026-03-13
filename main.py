"""
Training pipeline for video classification on Kinetics.
Compatible with PyTorch 2.5+ with AMP, torchrun DDP, and CosineAnnealingLR.
"""

import os
import json
import time
import logging
import argparse
import datetime
import torch
import torch.nn as nn
from utils import util
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

from models.model3D import load_model
from dataset.kinetics import get_training_set, get_validation_set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



# Train epoch

def train_model(
    epoch: int,
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    opt,
    epoch_logger: util.Logger,
    batch_logger: util.Logger,
    is_main: bool = True,
):
    if is_main:
        logger.info(f"Training epoch {epoch}")

    model.train()

    batch_time = util.AverageMeter()
    data_time  = util.AverageMeter()
    losses     = util.AverageMeter()
    top1       = util.AverageMeter()
    top5       = util.AverageMeter()

    device   = next(model.parameters()).device
    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=opt.amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        prec1, prec5 = util.calculate_accuracy(outputs.detach(), targets, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if opt.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if is_main and i % opt.log_interval == 0:
            logger.info(
                f"Epoch [{epoch}][{i}/{len(data_loader)}]  "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}  "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})  "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                f"Prec@1 {top1.val:.4f} ({top1.avg:.4f})  "
                f"Prec@5 {top5.val:.4f} ({top5.avg:.4f})"
            )

        if is_main:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter':  (epoch - 1) * len(data_loader) + (i + 1),
                'loss':  losses.val,
                'prec1': top1.val,
                'prec5': top5.val,
                'lr':    optimizer.param_groups[0]['lr'],
            })

    if is_main:
        epoch_logger.log({
            'epoch': epoch,
            'loss':  losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'lr':    optimizer.param_groups[0]['lr'],
        })



# Validation epoch

def eval_model(
    epoch: int,
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    opt,
    val_logger: util.Logger,
    rank: int = 0,
    world_size: int = 1,
):
    is_main = rank == 0
    if is_main:
        logger.info(f"Validation epoch {epoch}")

    model.eval()

    batch_time = util.AverageMeter()
    data_time  = util.AverageMeter()

    # Track sums and counts manually for correct all_reduce aggregation
    loss_sum  = 0.0
    top1_sum  = 0.0
    top5_sum  = 0.0
    n_samples = 0

    device   = next(model.parameters()).device
    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=opt.amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

            prec1, prec5 = util.calculate_accuracy(outputs, targets, topk=(1, 5))

            batch_n    = inputs.size(0)
            loss_sum  += loss.item()  * batch_n
            top1_sum  += prec1        * batch_n
            top5_sum  += prec5        * batch_n
            n_samples += batch_n

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if is_main and i % opt.log_interval == 0:
                logger.info(
                    f"Val [{epoch}][{i}/{len(data_loader)}]  "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})  "
                    f"Loss {loss.item():.4f}  "
                    f"Prec@1 {prec1:.4f}  "
                    f"Prec@5 {prec5:.4f}"
                )

    # --- All-reduce across ranks ---
    # Pack into a single tensor to minimize communication overhead
    if world_size > 1:
        stats = torch.tensor(
            [loss_sum, top1_sum, top5_sum, float(n_samples)],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, top1_sum, top5_sum, n_samples = stats.tolist()

    loss_avg = loss_sum / n_samples
    top1_avg = top1_sum / n_samples
    top5_avg = top5_sum / n_samples

    if is_main:
        val_logger.log({
            'epoch': epoch,
            'loss':  loss_avg,
            'prec1': top1_avg,
            'prec5': top5_avg,
        })

    return loss_avg, top1_avg



# Main

def main(opt):
    # torchrun sets LOCAL_RANK and WORLD_SIZE automatically
    # fallback to 0 and 1 for single GPU runs without torchrun
    rank       = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    is_distributed = world_size > 1
    is_main        = rank == 0

    # Suppress INFO logs on non-main ranks
    # Warnings and errors still pass through from all ranks
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    if is_distributed:
        # Set up DDP
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=20)
        )

    device = torch.device(f'cuda:{rank}')

    # ---- Model ----
    model, parameters = load_model(opt)
    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if is_main:
        logger.info(str(model))

    # ---- Criterion ----
    criterion = nn.CrossEntropyLoss().to(device)

    # ---- AMP Scaler ----
    scaler = GradScaler(enabled=opt.amp)

    # ---- Optimizer ----
    dampening = 0 if opt.nesterov else opt.dampening
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov,
    )

    # ---- Scheduler ----
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.n_epochs,
        eta_min=opt.learning_rate * 1e-3,
    )

    # ---- Resume ----
    best_prec1  = 0.0
    begin_epoch = opt.begin_epoch

    if opt.resume_path:
        if is_main:
            logger.info(f"Resuming from checkpoint: {opt.resume_path}")
        checkpoint  = torch.load(opt.resume_path, map_location=device, weights_only=True)
        assert opt.arch == checkpoint['arch'], "Architecture mismatch in checkpoint"
        if is_main:                              # ← only rank 0 needs this
            best_prec1 = checkpoint['best_prec1']
        begin_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        if is_main:
            logger.info(f"Resumed at epoch {begin_epoch}, best_prec1={best_prec1:.4f}")

    # ---- Training data loader (all ranks) ----
    training_data = get_training_set(opt)
    train_sampler = DistributedSampler(training_data, shuffle=True) if is_distributed else None
    train_loader  = DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        persistent_workers=opt.num_workers > 0, # True - avoiding re-fork overhead and importantly keeping _envs LMDB handles warm across epochs
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if opt.num_workers > 0 else None, # how many batches each worker prepares ahead of time in its own queue
    )

    # ---- Validation data loader (all ranks) ----
    val_data    = get_validation_set(opt)
    val_sampler = DistributedSampler(val_data, shuffle=False, drop_last=True) if is_distributed else None
    val_loader  = DataLoader(
        val_data,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        persistent_workers=opt.num_workers > 0,
        pin_memory=True,
        prefetch_factor=2 if opt.num_workers > 0 else None,
    )

    # ---- Loggers (main process only) ----
    train_logger = train_batch_logger = val_logger = None
    if is_main:
        train_logger = util.Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'],
        )
        train_batch_logger = util.Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'],
        )
        val_logger = util.Logger(
            os.path.join(opt.result_path, 'val.log'),
            ['epoch', 'loss', 'prec1', 'prec5'],
        )

    # ---- Training loop ----
    if is_main:
        logger.info("Starting training")

    for epoch in range(begin_epoch, opt.n_epochs + 1):

        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_model(
            epoch, train_loader, model, criterion,
            optimizer, scaler, opt,
            train_logger, train_batch_logger,
            is_main=is_main,
        )

        # Sync
        if is_distributed:
            dist.barrier()

        scheduler.step()

        # Validation runs on ALL ranks
        val_model = model.module if isinstance(model, DDP) else model

        _, prec1 = eval_model(
            epoch, val_loader, val_model, criterion,
            opt, val_logger,
            rank=rank,
            world_size=world_size,
        )

        # Only rank 0 decides best and saves
        if is_main:
            is_best    = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

        if is_main:
            state = {
                'epoch':      epoch,
                'arch':       opt.arch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'scaler':     scaler.state_dict(),
                'best_prec1': best_prec1,
            }
            util.save_checkpoint(state, is_best, opt)
            
        # Second barrier before next epoch starts
        if is_distributed:
            dist.barrier()

    if is_main:
        train_logger.close()
        train_batch_logger.close()
        val_logger.close()

    if is_distributed:
        dist.destroy_process_group()



# Entry point


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--root_path',      default='/mnt/HDD10TB/kinetics600_lmdb', type=str)
    parser.add_argument('--result_path',    default='results',             type=str)
    parser.add_argument('--resume_path',    default='',                    type=str)
    parser.add_argument('--pretrain_path',  default='',                    type=str)

    # Dataset
    parser.add_argument('--num_classes',     default=600,  type=int)
    parser.add_argument('--sample_duration', default=16,   type=int)
    parser.add_argument('--sampling_step', default=1,    type=int)
    parser.add_argument('--input_size', default=112,  type=int)

    # Model
    parser.add_argument('--model', default='shufflenetv2', type=str)
    parser.add_argument('--width_mult', default=2.0, type=float)

    # Training
    parser.add_argument('--batch_size',     default=20,   type=int)
    parser.add_argument('--n_epochs',       default=100,  type=int)
    parser.add_argument('--begin_epoch',    default=1,    type=int)
    parser.add_argument('--learning_rate',  default=0.01, type=float)
    parser.add_argument('--momentum',       default=0.9,  type=float)
    parser.add_argument('--dampening',      default=0.0,  type=float)
    parser.add_argument('--weight_decay',   default=1e-4, type=float)
    parser.add_argument('--clip_grad_norm', default=1.0,  type=float, help='Max gradient norm. 0 to disable.')
    parser.add_argument('--nesterov',       action='store_true', default=False)
    parser.add_argument('--manual_seed',    default=42,   type=int)

    # AMP / compile
    parser.add_argument('--amp', action='store_true', default=False,  help='Enable mixed precision')

    # Misc
    parser.add_argument('--num_workers',  default=8,  type=int)
    parser.add_argument('--log_interval', default=1000, type=int, help='Log every N batches')

    opt = parser.parse_args()

    # Derived fields
    opt.arch       = opt.model
    opt.store_name = f"kinetics_{opt.model}_{opt.width_mult}x_RGB_{opt.sample_duration}"

    os.makedirs(opt.result_path, exist_ok=True)

    # Save opts before DDP init — runs once in the launching process
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)

    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed_all(opt.manual_seed)

    main(opt)