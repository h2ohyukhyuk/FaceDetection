"""Original WIDER FACE train/val without converting the dataset on disk."""
from pathlib import Path
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from dataset.wider_face_center_train2 import WiderFaceCenterTrain
from utils.rect_transforms import ImageBoxSRTF, ImageBoxBG
from utils import transforms as transf_img


def read_annotations(path):
    samples=[]
    with open(path) as f:
        lines=iter(f)
        for line in lines:
            name=line.strip()
            if not name.endswith('.jpg'):
                continue  # WIDER zero-face annotations may include a dummy row.
            count=int(next(lines))
            labels=[list(map(float,next(lines).split())) for _ in range(count)]
            samples.append((name,labels))
    return samples


class OriginalTrain(WiderFaceCenterTrain):
    def __init__(self, root, transform_img, input_size):
        self.transform_img=transform_img
        self.src_rect_min=24
        self.dst_rect_min=10
        self.dst_height,self.dst_width=input_size
        if tuple(input_size)!=(256,384):
            raise ValueError('The existing heatmap target generator requires INPUT_SIZE=[256,384]')
        self.train_bg_prob=.2
        self.transform_srtf=ImageBoxSRTF(dst_width=self.dst_width,dst_height=self.dst_height,
                                       src_rect_min=self.src_rect_min,dst_rect_min=self.dst_rect_min)
        self.transform_bg=ImageBoxBG(dst_width=self.dst_width,dst_height=self.dst_height)
        records=read_annotations(Path(root)/'wider_face_split/wider_face_train_bbx_gt.txt')
        self.imgs_path=[str(Path(root)/'WIDER_train/images'/name) for name,_ in records]
        self.words=[[box for box in boxes if box[7]==0 and box[2]>0 and box[3]>0] for _,boxes in records]
        missing=[p for p in self.imgs_path if not Path(p).is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} WIDER training images missing; first: {missing[0]}')
        logging.info('WIDER train: %d images; original annotations, existing augmentation/target generation',len(self))


class OriginalValid(Dataset):
    def __init__(self, root, input_size, gt_dir):
        self.input_size=tuple(input_size)
        mat=loadmat(Path(root)/'wider_face_split/wider_face_val.mat')
        subsets={s:loadmat(Path(gt_dir)/f'wider_{s}_val.mat')['gt_list'] for s in ('easy','medium','hard')}
        self.samples=[]
        for i,event in enumerate(mat['event_list']):
            event=str(event[0][0])
            for j,item in enumerate(mat['file_list'][i][0]):
                name=event+'/'+str(item[0][0])+'.jpg'
                boxes=mat['face_bbx_list'][i][0][j][0].astype(np.float64).reshape(-1,4)
                keep={}
                keep['all']=~mat['invalid_label_list'][i][0][j][0].reshape(-1).astype(bool)
                keep['all'] &= (boxes[:,2]>0)&(boxes[:,3]>0)
                for s,gt in subsets.items():
                    mask=np.zeros(len(boxes),dtype=bool)
                    indices=gt[i][0][j][0].reshape(-1).astype(int)-1
                    if np.any(indices<0) or np.any(indices>=len(boxes)): raise ValueError('GT index mismatch')
                    mask[indices]=True;keep[s]=mask
                self.samples.append({'name':name,'path':str(Path(root)/'WIDER_val/images'/name),
                                     'boxes':boxes,'keep':keep})
        missing=[s['path'] for s in self.samples if not Path(s['path']).is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} WIDER validation images missing; first: {missing[0]}')
        logging.info('WIDER val: %d images; all original faces retained; input=%s, single scale',len(self),self.input_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self,index):
        sample=self.samples[index]
        img=cv2.imread(sample['path'])
        if img is None: raise RuntimeError(f"Cannot read {sample['path']}")
        h,w=img.shape[:2];dh,dw=self.input_size
        img=cv2.resize(img,(dw,dh))
        image=torch.from_numpy(np.ascontiguousarray(((img.astype(np.float32)-128)/255).transpose(2,0,1)))
        return {'image':image,'name':sample['name'],'boxes':sample['boxes'],
                'keep':sample['keep'],'scale':np.array([dw/w,dh/h],dtype=np.float64)}


def valid_collate(samples):
    return {'image':torch.stack([s['image'] for s in samples]),'samples':samples}


def get_train_loader(conf):
    augmentation=transf_img.Compose([transf_img.PixelNoise(noise_factor=.4),
                                    transf_img.Gamma(noise_factor=.3),transf_img.Blur(blur=.1)])
    dataset=OriginalTrain(conf['DATASET']['ROOT_TRAIN'],augmentation,conf['MODEL']['INPUT_SIZE'])
    c=conf['TRAIN']
    return DataLoader(dataset,batch_size=c['BATCH_SIZE'],shuffle=c['SHUFFLE'],num_workers=c['WORKERS'])


def get_valid_loader(conf):
    dataset=OriginalValid(conf['DATASET']['ROOT_VALID'],conf['MODEL']['INPUT_SIZE'],conf['VALID']['GT_DIR'])
    c=conf['VALID']
    return DataLoader(dataset,batch_size=c['BATCH_SIZE'],shuffle=False,num_workers=c['WORKERS'],collate_fn=valid_collate)
