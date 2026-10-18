"""Original WIDER FACE train/val without converting the dataset on disk."""
from pathlib import Path
import logging
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from dataset.wider_face_center_train2 import WiderFaceCenterTrain
from utils.rect_transforms import ImageBoxSRTF, ImageBoxBG
from utils import transforms as transf_img
from utils.augmentation_config import resolve_augmentation, input_hw


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
    def __init__(self, root, transform_img, input_size, augmentation_conf=None):
        self.transform_img=transform_img
        aug = resolve_augmentation(augmentation_conf)
        self.dst_height,self.dst_width = input_hw(input_size)
        self.stride = 4
        self.src_rect_min = aug['SRC_FACE_MIN']
        self.dst_rect_min = aug['FOREGROUND']['MIN_FACE_SIZE']
        self.min_target_face_size = min(input_size) * aug['MIN_FACE_RATIO']
        self.train_bg_prob = aug['BACKGROUND_PROB']
        self.fill_value = aug['FILL_VALUE']
        common = dict(dst_width=self.dst_width, dst_height=self.dst_height,
                      border_value=(self.fill_value,)*3,
                      translation_margin_ratio=aug['TRANSLATION_MARGIN_RATIO'],
                      translation_min_margin=aug['TRANSLATION_MIN_MARGIN'],
                      translation_max_margin_ratio=aug['TRANSLATION_MAX_MARGIN_RATIO'])
        fg, bg = aug['FOREGROUND'], aug['BACKGROUND']
        self.transform_srtf = ImageBoxSRTF(
            **common, src_rect_min=self.src_rect_min, dst_rect_min=self.dst_rect_min,
            scale_range=fg['SCALE_RANGE'], rot=fg['ROTATION_DEGREES'],
            rotation_prob=fg['ROTATION_PROB'], flip_prob=fg['FLIP_PROB'],
            crop_scale_range=fg['CROP_SCALE_RANGE'], min_scale_gap=fg['SCALE_MIN_GAP'])
        self.transform_bg = ImageBoxBG(
            **common, scale_range=bg['SCALE_RANGE'], rot=bg['ROTATION_DEGREES'],
            flip_prob=bg['FLIP_PROB'], crop_scale_range=bg['CROP_SCALE_RANGE'])
        records=read_annotations(Path(root)/'wider_face_split/wider_face_train_bbx_gt.txt')
        self.imgs_path=[str(Path(root)/'WIDER_train/images'/name) for name,_ in records]
        self.words=[[box for box in boxes if box[7]==0 and box[2]>0 and box[3]>0] for _,boxes in records]
        missing=[p for p in self.imgs_path if not Path(p).is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} WIDER training images missing; first: {missing[0]}')
        logging.info('WIDER train: %d images; input=%s; center candidates >=%gpx, blur=2 excluded; final targets >%.2f%% of short side (%.2fpx)',
                     len(self), input_size, self.src_rect_min, aug['MIN_FACE_RATIO']*100, self.min_target_face_size)


    def _erase_faces(self, image, boxes):
        """Used only for explicit background samples, never foreground filters."""
        height, width = image.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, x2 = np.clip([np.floor(x1), np.ceil(x2)], 0, width).astype(int)
            y1, y2 = np.clip([np.floor(y1), np.ceil(y2)], 0, height).astype(int)
            image[y1:y2, x1:x2] = self.fill_value

    def __getitem__(self, index):
        image = cv2.imread(self.imgs_path[index])
        if image is None:
            raise RuntimeError(f'Cannot read {self.imgs_path[index]}')
        height, width = image.shape[:2]
        labels = np.asarray(self.words[index], dtype=np.float32).reshape(-1, 10)
        boxes = labels[:, :4].copy()
        boxes[:, 2:] += boxes[:, :2]
        intersects = ((boxes[:, 0] < width) & (boxes[:, 1] < height) &
                      (boxes[:, 2] > 0) & (boxes[:, 3] > 0))
        boxes, labels = boxes[intersects], labels[intersects]
        sizes = np.max(boxes[:, 2:] - boxes[:, :2], axis=1)
        # Keep all faces through the transform, including non-candidates.
        # Heavy blur is never chosen as the anchor for foreground augmentation.
        candidates = np.flatnonzero((sizes >= self.src_rect_min) & (labels[:, 4] != 2))
        if random.random() < self.train_bg_prob:
            image, _ = self.transform_bg(image, boxes)
            target_boxes = np.empty((0, 4), np.float32)
        elif not len(candidates):
            # Preserve the original no-candidate background training policy.
            self._erase_faces(image, boxes)
            image = self.transform_srtf.random_crop(image)
            target_boxes = np.empty((0, 4), np.float32)
        else:
            matrix = self.transform_srtf.get_params(boxes[random.choice(candidates)])
            image = cv2.warpAffine(
                image, matrix, (self.dst_width, self.dst_height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=self.transform_srtf.border_value)
            boxes = self.transform_srtf.transform_rects(matrix, boxes)
            if random.random() < self.transform_srtf.flip_prob:
                image = cv2.flip(image, 1)
                boxes = self.transform_srtf.flip_rects(boxes, self.dst_width)
            sizes = np.max(boxes[:, 2:] - boxes[:, :2], axis=1)
            centers = (boxes[:, :2] + boxes[:, 2:]) * .5
            keep = ((labels[:, 4] != 2) & (sizes > self.min_target_face_size) &
                    (centers[:, 0] >= 0) & (centers[:, 0] < self.dst_width) &
                    (centers[:, 1] >= 0) & (centers[:, 1] < self.dst_height))
            # No erasing or ignore mask: excluded faces are background targets.
            target_boxes = boxes[keep]
        image = self.transform_img(np.ascontiguousarray(image))
        head = np.zeros((5, self.dst_height//self.stride, self.dst_width//self.stride), np.float32)
        mask = np.zeros((1, self.dst_height//self.stride, self.dst_width//self.stride), bool)
        for x1, y1, x2, y2 in target_boxes:
            nx, ny = (x1+x2)/(2*self.stride), (y1+y2)/(2*self.stride)
            ix, iy = int(nx), int(ny)
            self.heatmap_rect5x5(head[0], iy, ix)
            self.offset_rect5x5(head, mask, ix, iy, nx, ny, x2-x1, y2-y1, self.stride)
        image = ((image.astype(np.float32)-128)/255).transpose(2, 0, 1)
        return {'image': torch.from_numpy(np.ascontiguousarray(image)),
                'heads': [torch.from_numpy(head)], 'masks': [torch.from_numpy(mask)]}


class OriginalValid(Dataset):
    def __init__(self, root, input_size, gt_dir):
        self.input_size=input_hw(input_size)
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
    # Keep IPC-shared image tensors out of metadata retained for epoch metrics.
    return {'image':torch.stack([s['image'] for s in samples]),
            'samples':[{k:v for k,v in s.items() if k != 'image'} for s in samples]}


def get_train_loader(conf):
    aug = resolve_augmentation(conf.get('AUGMENTATION'))
    augmentation=transf_img.Compose([transf_img.PixelNoise(noise_factor=aug['PIXEL_NOISE']),
                                    transf_img.Gamma(noise_factor=aug['GAMMA']),
                                    transf_img.Blur(blur=aug['BLUR_PROB'], kernel_range=aug['BLUR_KERNEL_RANGE'])])
    dataset=OriginalTrain(conf['DATASET']['ROOT_TRAIN'],augmentation,conf['MODEL']['INPUT_SIZE'],aug)
    c=conf['TRAIN']
    return DataLoader(dataset,batch_size=c['BATCH_SIZE'],shuffle=c['SHUFFLE'],num_workers=c['WORKERS'])


def get_valid_loader(conf):
    dataset=OriginalValid(conf['DATASET']['ROOT_VALID'],conf['MODEL']['INPUT_SIZE'],conf['VALID']['GT_DIR'])
    c=conf['VALID']
    return DataLoader(dataset,batch_size=c['BATCH_SIZE'],shuffle=False,num_workers=c['WORKERS'],collate_fn=valid_collate)
