"""RetinaFace-style WIDER label.txt with coupled box/landmark augmentation."""
from pathlib import Path
import logging,random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from dataset.wider_face_original2 import OriginalTrain,read_annotations,get_valid_loader
from utils import transforms as transf_img
from utils.rect_transforms import ImageBoxSRTF,ImageBoxBG

FLIP_ORDER=[1,0,2,4,3]

def affine_landmarks(points,valid,matrix,width,height,flip=False):
    points=points.copy();valid=valid.copy()
    points=np.concatenate((points,np.ones((*points.shape[:-1],1))),axis=-1)@matrix.T
    if flip:
        points[:,:,0]=width-1-points[:,:,0]
        points=points[:,FLIP_ORDER];valid=valid[:,FLIP_ORDER]
    valid &= np.isfinite(points).all(axis=-1)
    valid &= (points[:,:,0]>=0)&(points[:,:,0]<width)&(points[:,:,1]>=0)&(points[:,:,1]<height)
    points[~valid]=0
    return points.astype(np.float32),valid

def landmark_targets(boxes, points, valid, height=64, width=96, stride=4):
    """Five point heatmaps and local xy offsets, with unknown labels ignored.

    Missing points ignore their face region in that point's heatmap. Positive
    targets from other faces override ignore regions. Colliding offset windows
    belong to the nearest integer point center, preserving distinct peaks.
    """
    heat = np.zeros((5, height, width), np.float32)
    heat_mask = np.ones_like(heat, dtype=bool)
    offsets = np.zeros((10, height, width), np.float32)
    mask = np.zeros_like(offsets, dtype=bool)
    distance = np.full((5, height, width), np.inf, np.float32)
    for box, annotated in zip(boxes, valid):
        x1, y1 = np.floor(box[:2] / stride).astype(int)
        x2, y2 = np.ceil(box[2:] / stride).astype(int)
        for k in np.flatnonzero(~annotated):
            heat_mask[k, np.clip(y1, 0, height):np.clip(y2, 0, height), np.clip(x1, 0, width):np.clip(x2, 0, width)] = False
    for face_points, annotated in zip(points, valid):
        for k in np.flatnonzero(annotated):
            nx, ny = face_points[k] / stride
            if not (np.isfinite(nx) and np.isfinite(ny) and 0 <= nx < width and 0 <= ny < height):
                continue
            ix, iy = int(nx), int(ny)
            OriginalTrain.heatmap_rect5x5(heat[k], iy, ix)
            for y in range(max(0, iy-2), min(height, iy+3)):
                for x in range(max(0, ix-2), min(width, ix+3)):
                    d = (x-ix)**2 + (y-iy)**2
                    if d < distance[k, y, x]:
                        distance[k, y, x] = d
                        offsets[2*k:2*k+2, y, x] = (nx-x, ny-y)
                        mask[2*k:2*k+2, y, x] = True
    heat_mask |= heat > 0
    return heat, heat_mask, offsets, mask

class LandmarkTrain(OriginalTrain):
    def __init__(self,conf,augmentation):
        root=Path(conf['DATASET']['ROOT_TRAIN'])
        self.dst_height,self.dst_width=conf['MODEL']['INPUT_SIZE']
        if (self.dst_height,self.dst_width)!=(256,384):raise ValueError('Existing targets require 256x384')
        self.transform_img=augmentation;self.src_rect_min=24;self.dst_rect_min=10;self.train_bg_prob=.2
        self.transform_srtf=ImageBoxSRTF(dst_width=self.dst_width,dst_height=self.dst_height,src_rect_min=self.src_rect_min,dst_rect_min=self.dst_rect_min)
        self.transform_bg=ImageBoxBG(dst_width=self.dst_width,dst_height=self.dst_height)
        # Keep the official training split and exclude known invalid boxes.
        official=dict(read_annotations(root/'wider_face_split/wider_face_train_bbx_gt.txt'))
        records={};name=None
        with open(conf['DATASET']['LANDMARK_LABEL']) as f:
            for line in f:
                if line.startswith('#'):
                    name=line[1:].strip()
                    if name in records:raise ValueError(f'Duplicate image: {name}')
                    records[name]=[]
                elif line.strip():
                    row=list(map(float,line.split()))
                    if name is None or len(row)!=20:raise ValueError('Expected bbox + 5*(x,y,flag) + confidence')
                    records[name].append(row)
        missing=set(official)-set(records)
        if missing:raise ValueError(f'Landmark label missing {len(missing)} train images')
        self.samples=[]
        self.blur_boxes=[]
        for name,boxes in official.items():
            # Official WIDER attributes: xywh, blur, expression, illumination,
            # invalid, occlusion, pose. Landmark label.txt has no blur field.
            blurred=[b[:4] for b in boxes if b[4]==2 and b[2]>0 and b[3]>0]
            invalid={tuple(b[:4]) for b in boxes if b[7]!=0 or b[4]==2}
            rows=[r for r in records[name] if r[2]>0 and r[3]>0 and tuple(r[:4]) not in invalid]
            path=root/'WIDER_train/images'/name
            if not path.is_file():raise FileNotFoundError(path)
            self.samples.append((str(path),np.asarray(rows,dtype=np.float32).reshape(-1,20)))
            self.blur_boxes.append(np.asarray(blurred,dtype=np.float32).reshape(-1,4))
        logging.info('WIDER blur=2: %d face boxes filled with 128 before augmentation and excluded from all targets',
                     sum(len(boxes) for boxes in self.blur_boxes))
        logging.info('WIDER landmark train: %d images; invalid points masked; flags 0/1 annotated, -1 missing',len(self))

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        path,rows=self.samples[index];im=cv2.imread(path)
        if im is None:raise RuntimeError(f'Cannot read {path}')
        h,w=im.shape[:2];rows=rows.copy()
        # Erase before both foreground and background augmentation. Clip both
        # ends so boxes entirely outside the image cannot create negative slices.
        for x,y,bw,bh in self.blur_boxes[index]:
            x1,x2=np.clip([np.floor(x),np.ceil(x+bw)],0,w).astype(int)
            y1,y2=np.clip([np.floor(y),np.ceil(y+bh)],0,h).astype(int)
            im[y1:y2,x1:x2]=128
        good=(rows[:,0]<w)&(rows[:,1]<h)&(rows[:,0]+rows[:,2]>0)&(rows[:,1]+rows[:,3]>0)
        rows=rows[good];boxes=rows[:,:4].copy();boxes[:,2:]+=boxes[:,:2]
        lm=rows[:,4:19].reshape(-1,5,3)
        points=lm[:,:,:2].copy()
        valid=(lm[:,:,2]>=0)&(points>=0).all(axis=-1)&np.isfinite(points).all(axis=-1)
        if random.random()<self.train_bg_prob:
            im,boxes=self.transform_bg(im,boxes)
            # Existing background augmenter has no point correspondence mapping.
            points=np.zeros((len(boxes),5,2),np.float32);valid=np.zeros((len(boxes),5),bool)
        else:
            size=np.maximum(boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1])
            chosen=size>=self.src_rect_min
            for b in boxes[~chosen]:
                x1,y1,x2,y2=b.astype(int);im[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=128
            boxes=boxes[chosen];points=points[chosen];valid=valid[chosen]
            if len(boxes):
                matrix=self.transform_srtf.get_params(boxes[random.randrange(len(boxes))])
                im=cv2.warpAffine(im,matrix,(self.dst_width,self.dst_height),borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
                boxes=self.transform_srtf.transform_rects(matrix,boxes)
                flip=random.random()<self.transform_srtf.flip_prob
                points,valid=affine_landmarks(points,valid,matrix,self.dst_width,self.dst_height,flip)
                if flip:
                    im=cv2.flip(im,1)
                    # Pixel-center convention matches the existing box flip.
                    boxes=self.transform_srtf.flip_rects(boxes,self.dst_width)
            else:
                im=self.transform_srtf.random_crop(im)
        retained=[]
        for i,b in enumerate(boxes):
            x1,y1,x2,y2=b;cx=(x1+x2)/2;cy=(y1+y2)/2
            if max(x2-x1,y2-y1)<self.dst_rect_min or not (0<=cx<self.dst_width and 0<=cy<self.dst_height):
                ix1,ix2=np.clip([x1,x2],0,self.dst_width).astype(int)
                iy1,iy2=np.clip([y1,y2],0,self.dst_height).astype(int)
                im[iy1:iy2,ix1:ix2]=128
            else:retained.append(i)
        im=self.transform_img(np.ascontiguousarray(im))
        head=np.zeros((5,64,96),np.float32);mask=np.zeros((1,64,96),bool)
        for i in retained:
            x1,y1,x2,y2=boxes[i];bw=x2-x1;bh=y2-y1;cx=(x1+x2)/2;cy=(y1+y2)/2
            nx,ny=cx/4,cy/4;ix,iy=int(nx),int(ny)
            self.heatmap_rect5x5(head[0],iy,ix)
            self.offset_rect5x5(head,mask,ix,iy,nx,ny,bw,bh,4.)
        lm_heat,lm_heat_mask,target,lm_mask=landmark_targets(
            boxes[retained],points[retained],valid[retained])
        image=((im.astype(np.float32)-128)/255).transpose(2,0,1)
        return {'image':torch.from_numpy(np.ascontiguousarray(image)),
                'heads':[torch.from_numpy(head)],'masks':[torch.from_numpy(mask)],
                'landmarks':torch.from_numpy(target),'landmark_mask':torch.from_numpy(lm_mask),
                'landmark_heatmaps':torch.from_numpy(lm_heat),
                'landmark_heatmap_mask':torch.from_numpy(lm_heat_mask)}

def get_train_loader(conf):
    augmentation=transf_img.Compose([transf_img.PixelNoise(noise_factor=.4),transf_img.Gamma(noise_factor=.3),transf_img.Blur(blur=.1)])
    dataset=LandmarkTrain(conf,augmentation);c=conf['TRAIN']
    return DataLoader(dataset,batch_size=c['BATCH_SIZE'],shuffle=c['SHUFFLE'],num_workers=c['WORKERS'])
