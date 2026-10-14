"""WIDER matching (inclusive pixel IoU, ignored GT) and 1000-point AP.

AP follows WIDER's normalized-score threshold protocol; operating-point
precision/recall/F1 use original confidence scores and all valid GT.
"""
import numpy as np


def match_image(pred, boxes, keep, iou=.5):
    pred=np.asarray(pred,dtype=np.float64).reshape(-1,5)
    pred=pred[np.argsort(-pred[:,4],kind='stable')]
    tp=np.zeros(len(pred),dtype=float);proposal=np.ones(len(pred),dtype=float)
    if not len(boxes): return pred,tp,proposal
    gt=boxes.copy();gt[:,2:]+=gt[:,:2]
    area=(gt[:,2]-gt[:,0]+1)*(gt[:,3]-gt[:,1]+1)
    found=np.zeros(len(gt),dtype=bool)
    for i,p in enumerate(pred):
        wh=np.maximum(0,np.minimum(p[2:4],gt[:,2:])-np.maximum(p[:2],gt[:,:2])+1)
        inter=wh[:,0]*wh[:,1]
        pa=(p[2]-p[0]+1)*(p[3]-p[1]+1)
        overlap=inter/np.maximum(pa+area-inter,1e-12)
        j=int(np.argmax(overlap))
        if overlap[j]>=iou:
            if not keep[j]: proposal[i]=0
            elif not found[j]: tp[i]=1;found[j]=True
    return pred,tp,proposal


def voc_ap(recall,precision):
    r=np.r_[0.,recall,1.];p=np.r_[0.,precision,0.]
    p=np.maximum.accumulate(p[::-1])[::-1]
    i=np.flatnonzero(r[1:]!=r[:-1])
    return float(np.sum((r[i+1]-r[i])*p[i+1]))


def evaluate(records,score_threshold=.4,iou=.5):
    scores=[p[:,4] for _,p in records if len(p)]
    lo=min(float(s.min()) for s in scores) if scores else 0.
    hi=max(float(s.max()) for s in scores) if scores else 1.
    thresholds=1-(np.arange(1000)+1)/1000
    metrics={}
    for subset in ('all','easy','medium','hard'):
        curve=np.zeros((1000,2));total=0;op_tp=op_proposals=0
        for sample,pred in records:
            keep=sample['keep'][subset];total+=int(keep.sum())
            p,tp,proposal=match_image(pred,sample['boxes'],keep,iou)
            if not len(p): continue
            scores_norm=(p[:,4]-lo)/(hi-lo) if hi>lo else np.ones(len(p))
            counts=np.searchsorted(-scores_norm,-thresholds,side='right')
            cum_tp=np.r_[0,np.cumsum(tp)];cum_prop=np.r_[0,np.cumsum(proposal)]
            curve[:,0]+=cum_prop[counts];curve[:,1]+=cum_tp[counts]
            select=p[:,4]>=score_threshold
            op_tp+=int(tp[select].sum());op_proposals+=int(proposal[select].sum())
        recall=curve[:,1]/max(total,1)
        precision=np.divide(curve[:,1],curve[:,0],out=np.zeros(1000),where=curve[:,0]>0)
        pr=op_tp/max(op_proposals,1);re=op_tp/max(total,1)
        metrics[subset]={'AP50':voc_ap(recall,precision),'precision':pr,'recall':re,
                         'F1':2*pr*re/max(pr+re,1e-12),'TP':op_tp,
                         'FP':op_proposals-op_tp,'FN':total-op_tp,'GT':total}
    return metrics
