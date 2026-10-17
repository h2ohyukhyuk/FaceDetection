"""Run the original FDDB evaluator on rectangle predictions and ellipse GT."""
from pathlib import Path
import json,logging,subprocess
import numpy as np
import torch
from dataset.fddb import get_loader
from utils.decode_output import decode_heads,nms

class FDDBValidation:
    def __init__(self,conf):
        self.conf=conf;self.loader=get_loader(conf)
        self.device=torch.device('cuda:0' if conf['DEVICE']=='GPU' else 'cpu')
        self.binary=Path(conf['FDDB']['EVALUATOR'])
        if not self.binary.is_file():raise FileNotFoundError(f'FDDB evaluator missing: {self.binary}')
    def __call__(self,model,epoch):
        if epoch%self.conf['FDDB'].get('INTERVAL',1):return
        out=Path(self.conf['MODEL']['PATH_SAVE'])/'fddb'/f'{epoch:03d}'
        out.mkdir(parents=True,exist_ok=True);model.eval();records=[]
        v=self.conf['VALID']
        with torch.no_grad():
            for i,b in enumerate(self.loader):
                y=model(b['image'].to(self.device,dtype=torch.float32)).float()
                if not y.isfinite().all():raise FloatingPointError('Nonfinite FDDB output')
                y[:,0:1]=y[:,0:1].sigmoid()
                boxes=decode_heads([(y.cpu().numpy(),4)],score_thresh=v.get('AP_SCORE_THRESHOLD',.001))
                for s,p in zip(b['samples'],boxes):
                    p=nms(p,iou_thresh=v.get('NMS_IOU_THRESHOLD',.4)).astype(np.float64)
                    p=p[np.isfinite(p).all(axis=1)]
                    p=p[(p[:,2]>p[:,0])&(p[:,3]>p[:,1])]
                    sx,sy=s['scale'];p[:,[0,2]]/=sx;p[:,[1,3]]/=sy
                    metadata={k:s[k] for k in ('name','fold','ellipses')}
                    records.append((metadata,p))
                if (i+1)%100==0:logging.info('FDDB epoch %03d: %d/%d images',epoch,len(records),len(self.loader.dataset))
        for fold in range(1,11):
            self.write_predictions(out/f'fold-{fold:02d}-out.txt',[(s,p) for s,p in records if s['fold']==fold])
        self.write_predictions(out/'detections.txt',records)
        (out/'images.txt').write_text(''.join(s['name']+'\n' for s,p in records))
        with open(out/'ellipses.txt','w') as f:
            for s,p in records:
                f.write(f"{s['name']}\n{len(s['ellipses'])}\n")
                for e in s['ellipses']:f.write(' '.join(map(str,e))+' 1\n')
        cmd=[str(self.binary.resolve()),'-a',str((out/'ellipses.txt').resolve()),'-d',str((out/'detections.txt').resolve()),
             '-f','0','-i',str((Path(self.conf['FDDB']['ROOT'])/'originalPics').resolve())+'/',
             '-l',str((out/'images.txt').resolve()),'-r',str((out/'roc').resolve()),'-z','.jpg']
        r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/'evaluator.log').write_text(r.stdout)
        if r.returncode:raise RuntimeError(f'FDDB evaluator failed; see {out}/evaluator.log')
        metrics={'images':len(records),'faces':sum(len(s['ellipses']) for s,p in records),'protocol':'FDDB original ellipse GT vs rectangle detections; pooled 10 folds'}
        for mode in ['Disc','Cont']:
            curve=np.loadtxt(out/f'roc{mode}ROC.txt',ndmin=2)
            if curve.size==0:raise RuntimeError('FDDB evaluator produced empty ROC')
            for fp in [100,500,1000]:
                allowed=curve[:,1]<=fp
                metrics[f'{mode}_TPR_at_FP{fp}']=float(curve[allowed,0].max()) if allowed.any() else 0.
        logging.info('FDDB epoch %03d | discrete TPR @ FP100=%.6f FP500=%.6f FP1000=%.6f | continuous @ FP1000=%.6f | images=%d faces=%d',
                     epoch,metrics['Disc_TPR_at_FP100'],metrics['Disc_TPR_at_FP500'],metrics['Disc_TPR_at_FP1000'],metrics['Cont_TPR_at_FP1000'],metrics['images'],metrics['faces'])
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        return metrics
    @staticmethod
    def write_predictions(path,records):
        with open(path,'w') as f:
            for s,p in records:
                f.write(f"{s['name']}\n{len(p)}\n")
                for x1,y1,x2,y2,score in p:f.write(f'{x1:.8f} {y1:.8f} {x2-x1:.8f} {y2-y1:.8f} {score:.8f}\n')
