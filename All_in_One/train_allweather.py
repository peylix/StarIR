"""Train StarIR on the mixed AllWeather set (Outdoor-Rain + RainDrop + Snow100K).

Follows the paper's all-in-one protocol: no validation split — the full
training set (including any images whose gt sits in gt_val/) is trained on,
and checkpoints are evaluated afterwards with test_allweather.py.

Example:
    python train_allweather.py --data_dir ~/autodl-tmp/allweather --num_gpus 1
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from net.model import StarIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.allweather_utils import AllWeatherTrainDataset


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)

        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class AllWeatherModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.net = StarIR()
        self.loss_fn = nn.L1Loss()
        self.loss_fft = FFTLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        loss += self.loss_fft(restored, clean_patch)
        self.log('train_loss', loss)
        return loss

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=self.opt.warmup_epochs,
            max_epochs=self.opt.epochs)
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True,
                        help='root of the AllWeather training set (contains input/, gt/)')
    parser.add_argument('--input_subdir', type=str, default='input')
    parser.add_argument('--gt_subdir', type=str, default='gt')
    parser.add_argument('--extra_gt_subdir', type=str, default='gt_val',
                        help='secondary gt folder, also used for training')

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size per GPU')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='gradient accumulation steps; effective batch = '
                             'batch_size * num_gpus * this')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--ckpt_dir', type=str, default='ckpt/allweather')
    parser.add_argument('--ckpt_every', type=int, default=10,
                        help='save a checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of a .ckpt file to resume from')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    opt = parser.parse_args()
    pl.seed_everything(opt.seed, workers=True)

    trainset = AllWeatherTrainDataset(
        opt.data_dir, patch_size=opt.patch_size, input_subdir=opt.input_subdir,
        gt_subdir=opt.gt_subdir, extra_gt_subdir=opt.extra_gt_subdir)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True,
                             shuffle=True, drop_last=True, num_workers=opt.num_workers)

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, save_last=True,
                                          every_n_epochs=opt.ckpt_every, save_top_k=-1,
                                          filename='epoch{epoch:03d}',
                                          auto_insert_metric_name=False)

    logger = TensorBoardLogger(save_dir='logs/', name='StarIR-AllWeather')
    model = AllWeatherModel(opt)
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator='gpu', devices=opt.num_gpus,
                         strategy='ddp_find_unused_parameters_true' if opt.num_gpus > 1 else 'auto',
                         accumulate_grad_batches=opt.accumulate_grad_batches,
                         logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=opt.resume)


if __name__ == '__main__':
    main()
