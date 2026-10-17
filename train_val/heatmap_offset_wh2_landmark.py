import torch
import torch.nn.functional as F

import time
import os
import logging
import numpy as np

from utils.train_utils import AverageMeter
from utils.model_utils import freeze_model
from utils.decode_output import decode_heads, nms, calc_metric


from train_val.heatmap_offset_wh2_widerface import Validationer


class Trainer():
    def __init__(self, conf, dataloader, optimizer, scheduler):
        self.conf = conf
        self.train_step = 0
        self.epoch = 1
        self.max_step = conf['TRAIN']['MAX_STEP']
        self.end_epoch = conf['TRAIN']['END_EPOCH']
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda:0" if conf['DEVICE'] == 'GPU' else "cpu")
        self.report_freq = conf['TRAIN']['PRINT_FREQ']

        self.sch_name = conf['OPTIMIZER']['LR_SCHEDULER']['SCHEDULER_NAME']

    def __call__(self, model, epoch):
        batch_time = AverageMeter(name="batch_time")
        data_time = AverageMeter(name="data_time")

        self.epoch = epoch
        model.train()
        model = freeze_model(self.conf, model, printInfo=False)

        st_data_load = time.time()
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.dataloader):
            data_time.update(time.time() - st_data_load)

            imgs = batch["image"].to(self.device, dtype=torch.float32)
            gt = [x.to(self.device, dtype=torch.float32) for x in batch["heads"]][0]
            mask = [x.to(self.device) for x in batch["masks"]][0]

            pred = model(imgs)

            loss_heat, loss_offset, loss_size = 0.0, 0.0, 0.0
            eps = 1e-6

            # Heatmap
            loss_heat = F.binary_cross_entropy_with_logits(
                pred[:, 0:1],
                gt[:, 0:1],
                reduction="mean"
            )

            # Offset
            loss_offset = F.smooth_l1_loss(
                pred[:, 1:3],
                gt[:, 1:3],
                reduction="none",
                beta=1.0,
            )
            loss_offset = loss_offset.sum(1, keepdim=True)
            loss_offset = (loss_offset * mask.float()).sum() / (mask.float().sum() + eps)

            # Size
            loss_size = F.smooth_l1_loss(
                pred[:, 3:5],
                gt[:, 3:5],
                reduction="none",
                beta=1.0,
            )
            loss_size = loss_size.sum(1, keepdim=True)
            loss_size = (loss_size * mask.float()).sum() / (mask.float().sum() + eps)

            landmark_gt = batch['landmarks'].to(self.device, dtype=torch.float32)
            landmark_mask = batch['landmark_mask'].to(self.device, dtype=torch.float32)
            landmark_error = F.smooth_l1_loss(pred[:, 10:20], landmark_gt, reduction='none', beta=1.0)
            loss_landmark_offset = (landmark_error * landmark_mask).sum() / landmark_mask.sum().clamp_min(1.0)
            landmark_heat = batch['landmark_heatmaps'].to(self.device, dtype=torch.float32)
            landmark_heat_mask = batch['landmark_heatmap_mask'].to(self.device, dtype=torch.float32)
            heat_error = F.binary_cross_entropy_with_logits(pred[:, 5:10], landmark_heat, reduction='none')
            loss_landmark_heat = (heat_error * landmark_heat_mask).sum() / landmark_heat_mask.sum().clamp_min(1.0)
            landmark_conf = self.conf['LANDMARK']
            landmark_weight = float(landmark_conf.get('LOSS_WEIGHT', 1.0))
            loss_landmark = (float(landmark_conf.get('HEATMAP_LOSS_WEIGHT', 1.0)) * loss_landmark_heat +
                             float(landmark_conf.get('OFFSET_LOSS_WEIGHT', 1.0)) * loss_landmark_offset)

            loss = (
                    1.0 * loss_heat +
                    1.0 * loss_offset +
                    0.2 * loss_size +
                    landmark_weight * loss_landmark
            )

            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite detection/landmark training loss')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.sch_name != 'multisteplr':
                self.scheduler.step()

            batch_time.update(time.time() - st_data_load)

            loss_item = loss.item()

            lr = self.optimizer.param_groups[0]['lr']
            self.report(loss_item, lr, data_time, batch_time)
            if self.train_step % self.report_freq == 0:
                logging.info('loss heat=%.6f offset=%.6f size=%.6f landmark_heat=%.6f landmark_offset=%.6f landmark_weight=%.3f supervised_point_targets=%d',
                             loss_heat.item(), loss_offset.item(), loss_size.item(), loss_landmark_heat.item(), loss_landmark_offset.item(),
                             landmark_weight, int(landmark_mask.sum().item() / 2))

            if self.max_step > 0 and i > self.max_step:
                break

            st_data_load = time.time()

        if self.sch_name == 'multisteplr':
            self.scheduler.step()

    def report(self, loss, lr, data_time, batch_time):
        self.train_step += 1
        if self.train_step % self.report_freq == 0:
            logging.info('train epoch %03d train_step %08d : loss: %0.6f lr: %0.6f data_load: %.6fs batch: %.6fs' %
                  (self.epoch, self.train_step, loss, lr, data_time.val, batch_time.val))

