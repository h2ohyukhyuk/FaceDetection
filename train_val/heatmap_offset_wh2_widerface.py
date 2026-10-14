"""Existing heatmap trainer with original WIDER FACE AP validation."""
from train_val.heatmap_offset_wh2 import Trainer
from utils.decode_output import decode_heads, nms
from utils.widerface_metrics import evaluate
import json
import logging
import time
from pathlib import Path
import numpy as np
import torch


class Validationer:
    def __init__(self,conf):
        self.conf=conf
        self.fddb = None
        if conf.get('FDDB', {}).get('ENABLED', False):
            from utils.fddb_validation import FDDBValidation
            self.fddb = FDDBValidation(conf)
        self.device=torch.device('cuda:0' if conf['DEVICE']=='GPU' else 'cpu')
        self.path=Path(conf['MODEL']['PATH_SAVE'])
        self.best_ap=-1.
        v=conf['VALID']
        self.score=float(v['SCORE_THRESHOLD'])
        self.decode_score=float(v.get('AP_SCORE_THRESHOLD',.001))
        self.nms_iou=float(v.get('NMS_IOU_THRESHOLD',.4))
        self.eval_iou=float(v.get('EVAL_IOU_THRESHOLD',.5))
        if not 0<=self.decode_score<=self.score<=1:
            raise ValueError('Require 0 <= AP_SCORE_THRESHOLD <= SCORE_THRESHOLD <= 1')
        if self.eval_iou!=.5: raise ValueError('WIDER AP50 requires EVAL_IOU_THRESHOLD=0.5')

    def __call__(self,model,epoch,dataloader):
        model.eval();records=[];start=time.perf_counter()
        with torch.no_grad():
            for i,batch in enumerate(dataloader):
                images=batch['image'].to(self.device,dtype=torch.float32)
                output=model(images).float()
                if not output.isfinite().all(): raise FloatingPointError('Non-finite validation model output')
                output[:,0:1]=output[:,0:1].sigmoid()
                decoded=decode_heads([(output.cpu().numpy(),4)],score_thresh=self.decode_score)
                for sample,boxes in zip(batch['samples'],decoded):
                    boxes=nms(boxes,iou_thresh=self.nms_iou).astype(np.float64)
                    boxes=boxes[np.isfinite(boxes).all(axis=1)]
                    boxes=boxes[(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])]
                    sx,sy=sample['scale']
                    boxes[:,[0,2]]/=sx;boxes[:,[1,3]]/=sy
                    records.append((sample,boxes))
                if (i+1)%100==0:
                    logging.info('WIDER validation epoch %03d: %d/%d images',epoch,len(records),len(dataloader.dataset))
        metrics=evaluate(records,self.score,self.eval_iou)
        saved=metrics['hard']['AP50']>self.best_ap
        self.path.mkdir(parents=True,exist_ok=True)
        if saved:
            self.best_ap=metrics['hard']['AP50']
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics},self.path/f'{epoch:03d}.pt')
        all_metrics=metrics['all']
        logging.info('WIDER val epoch %03d | AP50 Easy=%.6f Medium=%.6f Hard=%.6f All=%.6f | best Hard=%.6f%s',
                     epoch,metrics['easy']['AP50'],metrics['medium']['AP50'],metrics['hard']['AP50'],
                     all_metrics['AP50'],self.best_ap,' saved' if saved else '')
        logging.info('WIDER val epoch %03d | score>=%.3f IoU=%.2f | precision=%.6f recall=%.6f F1=%.6f | TP=%d FP=%d FN=%d | images=%d elapsed=%.1fs',
                     epoch,self.score,self.eval_iou,all_metrics['precision'],all_metrics['recall'],all_metrics['F1'],
                     all_metrics['TP'],all_metrics['FP'],all_metrics['FN'],len(records),time.perf_counter()-start)
        logging.info('WIDER protocol: single-scale input=%s AP min-score=%.4f NMS IoU=%.2f; All=valid GT, difficulty AP uses official subset indices',
                     self.conf['MODEL']['INPUT_SIZE'],self.decode_score,self.nms_iou)
        with open(self.path/'validation_metrics.jsonl','a') as f:
            f.write(json.dumps({'epoch':epoch,'images':len(records),'metrics':metrics,'saved':saved})+'\n')
        if self.fddb is not None:
            self.fddb(model, epoch)
        return metrics
