import torch
import torch.nn.functional as F

import time
import os
import logging
import numpy as np

from utils.train_utils import AverageMeter
from utils.model_utils import freeze_model
from utils.decode_output import decode_heads, nms, calc_metric


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
            head0, head1, head2, head3 = [x.to(self.device, dtype=torch.float32) for x in batch["heads"]]
            mask0, mask1, mask2, mask3 = [x.to(self.device) for x in batch["masks"]]

            ignore0, ignore1, ignore2, ignore3 = [x.to(self.device) for x in batch["ignores"] ]

            p_head0, p_head1, p_head2, p_head3 = model(imgs)

            preds = [p_head0, p_head1, p_head2, p_head3]
            gts = [head0, head1, head2, head3]
            masks = [mask0, mask1, mask2, mask3]
            ignores = [ignore0, ignore1, ignore2, ignore3]

            loss_heat, loss_offset, loss_size = 0.0, 0.0, 0.0
            eps = 1e-6

            for pred, gt, mask, ignore in zip(preds, gts, masks, ignores):

                heat_loss = F.binary_cross_entropy_with_logits(
                    pred[:, 0:1],
                    gt[:, 0:1],
                    reduction='none')
                # heatmap
                heat_loss = (heat_loss * ignore.float()).sum() / (ignore.float().sum() + eps)
                loss_heat += heat_loss
                # offset
                offset_loss = F.smooth_l1_loss( pred[:, 1:3], gt[:, 1:3], reduction='none')
                offset_loss = offset_loss.sum(1, keepdim=True)
                offset_loss = (offset_loss * mask.float()).sum() / (
                        mask.float().sum() + eps)
                loss_offset += offset_loss

                size_loss = F.smooth_l1_loss(pred[:, 3:5],gt[:, 3:5],reduction='none')
                size_loss = size_loss.sum(1, keepdim=True)
                size_loss = (size_loss * mask.float()).sum() / (
                        mask.float().sum() + eps
                )
                loss_size += size_loss

            loss = 1.0 * loss_heat + 1.0 * loss_offset + 0.2 * loss_size

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.sch_name != 'multisteplr':
                self.scheduler.step()

            batch_time.update(time.time() - st_data_load)

            loss_item = loss.item()

            lr = self.optimizer.param_groups[0]['lr']
            self.report(loss_item, lr, data_time, batch_time)

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

class Validationer():
    def __init__(self, conf):
        self.conf = conf
        self.epoch = 1
        self.local_path = conf['MODEL']['PATH_SAVE']
        self.max_step = conf['TRAIN']['MAX_STEP']
        self.end_epoch = conf['TRAIN']['END_EPOCH']
        self.device = torch.device("cuda:0" if conf['DEVICE'] == 'GPU' else "cpu")


        self.output_size = conf['MODEL']['OUTPUT_SIZE']

        self.score_th = conf["VALID"]["SCORE_THRESHOLD"]
        self.iou_th = conf["VALID"]["IOU_THRESHOLD"]

        self.best_f1 = 0.2

    def __call__(self, model, epoch, dataloader):

        self.epoch = epoch
        model.eval()

        total_tp = 0
        total_fp = 0
        total_fn = 0

        with torch.no_grad():

            for i, batch in enumerate(dataloader):

                imgs = batch["image"].to(self.device, dtype=torch.float32)
                rectss = batch["rects"]

                p_head0, p_head1, p_head2, p_head3 = model(imgs)

                p_head0[:, 0:1] = torch.sigmoid(p_head0[:, 0:1])
                p_head1[:, 0:1] = torch.sigmoid(p_head1[:, 0:1])
                p_head2[:, 0:1] = torch.sigmoid(p_head2[:, 0:1])
                p_head3[:, 0:1] = torch.sigmoid(p_head3[:, 0:1])

                heads = [
                    (p_head3.cpu().numpy(), 32),
                    (p_head2.cpu().numpy(), 16),
                    (p_head1.cpu().numpy(), 8),
                    (p_head0.cpu().numpy(), 4)]

                rects_batch = decode_heads(heads, score_thresh=self.score_th)
                rects_batch = [nms(rects, iou_thresh=self.iou_th) for rects in rects_batch ]

                for b in range(len(imgs)):

                    gt = []

                    rects = rectss[b].cpu().numpy()

                    for rect in rects:

                        e, x1, y1, x2, y2 = rect.astype(np.float32)

                        if e != 1:
                            break

                        gt.append([x1, y1, x2, y2])

                    gt = np.asarray(gt, np.float32)

                    pred = rects_batch[b]

                    tp, fp, fn = calc_metric(gt, pred, self.iou_th)

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        bSave = False
        if f1 > self.best_f1:
            self.best_f1 = f1
            bSave = True
            self.save(model)

        self.report(bSave, f1, precision, recall)

    def report(self, bSave, f1, precision, recall):

        msg = 'valid epoch: %03d saved ' % self.epoch if bSave else 'valid epoch: %03d ----- ' % self.epoch
        msg += f"f1: {f1}, precision: {precision}, recall: {recall}"
        logging.info(msg)

    def save(self, model):

        path_save = os.path.join(self.local_path, "%03d.pt" % self.epoch)
        state = {'model': model.state_dict()}
        torch.save(state, path_save)