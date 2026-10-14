"""FDDB original ellipse annotations and fixed-size inference loader."""
from pathlib import Path
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class FDDB(Dataset):
    def __init__(self,root,input_size):
        self.root=Path(root);self.input_size=tuple(input_size);self.samples=[]
        for fold in range(1,11):
            path=self.root/'FDDB-folds'/f'FDDB-fold-{fold:02d}-ellipseList.txt'
            with open(path) as f:
                lines=iter(f)
                for line in lines:
                    name=line.strip()
                    if not name:continue
                    n=int(next(lines));ellipses=np.array([list(map(float,next(lines).split()))[:5] for _ in range(n)]).reshape(-1,5)
                    self.samples.append({'name':name,'fold':fold,'ellipses':ellipses})
        missing=[s['name'] for s in self.samples if not (self.root/'originalPics'/(s['name']+'.jpg')).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} FDDB images missing: {missing[0]}')
        logging.info('FDDB: %d images / %d faces, 10 folds; validation only',len(self),sum(len(s['ellipses']) for s in self.samples))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i];p=self.root/'originalPics'/(s['name']+'.jpg')
        im=cv2.imread(str(p))
        if im is None:raise RuntimeError(f'Cannot read {p}')
        h,w=im.shape[:2];dh,dw=self.input_size
        im=cv2.resize(im,(dw,dh))
        tensor=torch.from_numpy(np.ascontiguousarray(((im.astype(np.float32)-128)/255).transpose(2,0,1)))
        return {**s,'image':tensor,'scale':np.array([dw/w,dh/h])}

def collate(samples):return {'image':torch.stack([s['image'] for s in samples]),'samples':samples}

def get_loader(conf):
    c=conf['FDDB'];dataset=FDDB(c['ROOT'],conf['MODEL']['INPUT_SIZE'])
    return DataLoader(dataset,batch_size=c.get('BATCH_SIZE',1),num_workers=c.get('WORKERS',1),shuffle=False,collate_fn=collate)
