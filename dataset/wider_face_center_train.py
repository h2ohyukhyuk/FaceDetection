import os
import os.path
import sys
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import cv2
import numpy as np
import random

from utils.rect_transforms import ImageBoxSRTF, ImageBoxBG
from utils import transforms as transf_img
from utils.heat_map import get_final_multi_preds

class WiderFaceCenterTrain(data.Dataset):
    def __init__(self, path_root, transform_img=None):
        self.transform_img = transform_img
        self.src_rect_min = 24 # src 크기 기준
        self.dst_rect_min= 10 # dst 크기 기준
        self.dst_width = 384
        self.dst_height = 256
        self.train_bg_prob = 0.2
        self.transform_srtf = ImageBoxSRTF(dst_width=self.dst_width,
                                           dst_height=self.dst_height,
                                           src_rect_min=self.src_rect_min,
                                           dst_rect_min=self.dst_rect_min)
        self.transform_bg = ImageBoxBG(dst_width=self.dst_width,
                                           dst_height=self.dst_height)

        self.imgs_path = []
        self.words = []
        path_label = os.path.join(path_root, "label.txt")
        f = open(path_label,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = path_label.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)


    def __len__(self):
        return len(self.imgs_path)

    @staticmethod
    def heatmap_rect5x5(map, iy, ix):
        h, w = map.shape

        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):

                y = iy + dy
                x = ix + dx

                if 0 <= y < h and 0 <= x < w:
                    if dy == 0 and dx == 0:
                        map[y, x] = max(map[y, x], 1.0)
                    elif abs(dy) == 2 or abs(dx) == 2:
                        map[y, x] = max(map[y, x], 0.2)
                    else:
                        map[y, x] = max(map[y, x], 0.6)


    @staticmethod
    def offset_rect5x5(head, mask, icx, icy, ncx, ncy, w, h, stride):
        """
        head : (5, H, W)
        mask   : (1, H, W) 또는 (H, W)
        """

        head_h, head_w = head.shape[1:]
        log_w, log_h = np.log(w / stride), np.log(h / stride)

        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):

                x = icx + dx
                y = icy + dy

                if x < 0 or x >= head_w or y < 0 or y >= head_h:
                    continue

                head[1, y, x] = ncx - x
                head[2, y, x] = ncy - y
                head[3, y, x] = log_w
                head[4, y, x] = log_h

                mask[0, y, x] = True


    @staticmethod
    def heatmap_ignore_rect5x5(ignore, iy, ix, value=False):
        """
        ignore : (H, W) bool
                 True  -> loss 계산
                 False -> ignore
        """

        _, h, w = ignore.shape

        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):

                y = iy + dy
                x = ix + dx

                if 0 <= y < h and 0 <= x < w:
                    ignore[0, y, x] = value



    def __getitem__(self, index):
        # print("index ", index)
        # index = 4082
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        rects = np.zeros((0,4), dtype=np.float32)

        for label in labels:
            x1, y1, w, h = label[:4]
            x2 = x1 + w  # x2
            y2 = y1 + h  # y2

            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height or w <= 0 or h <= 0:
                #print("outside face rect", x1, y1, x2, y2, self.imgs_path[index])
                pass
            else:
                rects = np.append(rects, np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4), axis=0)

        if random.random() < self.train_bg_prob:
            img_t, rects_t = self.transform_bg(img, rects)
        else:
            img_t, rects_t = self.transform_srtf(img, rects)

        rects_t_f = np.zeros((0, 4), dtype=np.float32)
        for rt_t in rects_t:
            x1, y1, x2, y2 = rt_t
            w_t, h_t = x2 - x1, y2 - y1
            cx_t, cy_t = (x1 + x2)*0.5, (y1 + y2)*0.5
            face_size = max(w_t, h_t)

            if face_size < self.dst_rect_min or cx_t < 0 or cx_t >= self.dst_width or cy_t < 0 or cy_t >= self.dst_height:
                x1 = int(max(0, min(self.dst_width, x1)))
                x2 = int(max(0, min(self.dst_width, x2)))
                y1 = int(max(0, min(self.dst_height, y1)))
                y2 = int(max(0, min(self.dst_height, y2)))
                img_t[y1:y2, x1:x2, :] = 128
            else:
                #rects_t_f = np.append(rects_t_f, np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4), axis=0)
                rects_t_f = np.append(rects_t_f, rt_t.reshape(1, 4), axis=0)

        img_t = self.transform_img(img_t)

        head1 = np.zeros(shape=(5, 64, 96), dtype=np.float32)
        head2 = np.zeros(shape=(5, 32, 48), dtype=np.float32)
        head3 = np.zeros(shape=(5, 16, 24), dtype=np.float32)
        head4 = np.zeros(shape=(5, 8, 12), dtype=np.float32)

        mask1 = np.zeros(shape=(1, 64, 96), dtype=bool)
        mask2 = np.zeros(shape=(1, 32, 48), dtype=bool)
        mask3 = np.zeros(shape=(1, 16, 24), dtype=bool)
        mask4 = np.zeros(shape=(1, 8, 12), dtype=bool)

        ignore1 = np.ones(shape=(1, 64, 96), dtype=bool)
        ignore2 = np.ones(shape=(1, 32, 48), dtype=bool)
        ignore3 = np.ones(shape=(1, 16, 24), dtype=bool)
        ignore4 = np.ones(shape=(1, 8, 12), dtype=bool)

        heads = [
            (head1, mask1, ignore1, 1, 4.0, self.dst_rect_min-1, 30),
            (head2, mask2, ignore2, 2, 8.0, 31, 79),
            (head3, mask3, ignore3, 3, 16.0, 80, 180),
            (head4, mask4, ignore4, 4, 32.0, 181, float("inf")),
        ]

        for rect in rects_t_f:

            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1

            face_size = max(w, h)

            rect_head_num = -100
            for _, _, _, head_num, _, min_size, max_size in heads:
                if min_size <= face_size <= max_size:
                    rect_head_num = head_num

            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            for head, mask, ignore, head_num, stride, min_size, max_size in heads:

                ncx = cx / stride
                ncy = cy / stride

                icx = int(ncx)
                icy = int(ncy)

                if min_size <= face_size <= max_size:
                    self.heatmap_rect5x5(head[0], icy, icx)
                    self.offset_rect5x5(head, mask, icx, icy, ncx, ncy, w, h, stride)
                elif abs(head_num - rect_head_num) == 1:
                    self.heatmap_ignore_rect5x5(ignore, icy, icx, value=False)

        for rect in rects_t_f:

            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1

            face_size = max(w, h)

            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5

            for head, mask, ignore, head_num, stride, min_size, max_size in heads:

                ncx = cx / stride
                ncy = cy / stride

                icx = int(ncx)
                icy = int(ncy)

                if min_size <= face_size <= max_size:
                    self.heatmap_ignore_rect5x5(ignore, icy, icx, value=True)

        # for rt in rects:
        #     x1, y1, x2, y2 = rt.astype(int)
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        #
        # cv2.imshow("",img)

        # vis_hm = visualize_heatmaps(head1, head2, head3, head4)
        # cv2.imshow("heatmaps", vis_hm)
        #
        # vis_mask = visualize_masks(mask1, mask2, mask3, mask4)
        # cv2.imshow("mask", vis_mask)
        #
        # vis_ignore = visualize_ignores(ignore1, ignore2, ignore3, ignore4)
        # cv2.imshow("ignore", vis_ignore)
        #
        # for rt_t_f in rects_t_f:
        #     x1, y1, x2, y2 = rt_t_f.astype(int)
        #     cv2.rectangle(img_t,(x1,y1),(x2,y2),(255,0,0),2)
        #
        # cv2.imshow("t", img_t)
        # cv2.waitKey(0)

        img_t = (img_t.astype(float) - 128.0) / 255.0
        img_tensor = torch.from_numpy(img_t.transpose((2, 0, 1))).float()
        head1_tensor = torch.from_numpy(head1).float()
        head2_tensor = torch.from_numpy(head2).float()
        head3_tensor = torch.from_numpy(head3).float()
        head4_tensor = torch.from_numpy(head4).float()
        mask1_tensor = torch.from_numpy(mask1)
        mask2_tensor = torch.from_numpy(mask2)
        mask3_tensor = torch.from_numpy(mask3)
        mask4_tensor = torch.from_numpy(mask4)
        ignore1_tensor = torch.from_numpy(ignore1)
        ignore2_tensor = torch.from_numpy(ignore2)
        ignore3_tensor = torch.from_numpy(ignore3)
        ignore4_tensor = torch.from_numpy(ignore4)

        return {
            "image": img_tensor,
            "heads": [
                head1_tensor,
                head2_tensor,
                head3_tensor,
                head4_tensor,
            ],
            "masks": [
                mask1_tensor,
                mask2_tensor,
                mask3_tensor,
                mask4_tensor,
            ],
            "ignores": [
                ignore1_tensor,
                ignore2_tensor,
                ignore3_tensor,
                ignore4_tensor,
            ],
        }


def visualize_heatmaps(head1, head2, head3, head4):
    heads = [head1, head2, head3, head4]
    max_h = max(head.shape[1] for head in heads)

    imgs = []

    for head in heads:

        hm = np.clip(head[0], 0, 1)
        hm = (hm * 255).astype(np.uint8)
        hm = cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR)

        # 빨간 테두리
        cv2.rectangle(
            hm,
            (0, 0),
            (hm.shape[1]-1, hm.shape[0]-1),
            (0, 0, 255),
            1
        )

        # 아래쪽 패딩
        pad = max_h - hm.shape[0]
        if pad > 0:
            hm = cv2.copyMakeBorder(
                hm,
                0, pad,
                0, 0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        imgs.append(hm)

    img_stack = np.hstack(imgs)
    return cv2.resize(img_stack,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)

def _visualize_binary_maps(map1, map2, map3, map4, scale=4):
    """
    map : (1,H,W) bool
    True -> 흰색
    False -> 검정
    """

    maps = [map1, map2, map3, map4]

    max_h = max(m.shape[1] for m in maps)
    imgs = []

    for m in maps:

        # (1,H,W) -> (H,W)
        img = np.where(m[0], 255, 0).astype(np.uint8)

        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1] - 1, img.shape[0] - 1),
            (0, 0, 255),
            1,
        )

        target_h = max_h * scale
        if img.shape[0] < target_h:
            img = cv2.copyMakeBorder(
                img,
                0,
                target_h - img.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        imgs.append(img)

    return np.hstack(imgs)


def visualize_masks(mask1, mask2, mask3, mask4, scale=4):
    """
    True(학습) = 흰색
    """
    return _visualize_binary_maps(
        mask1,
        mask2,
        mask3,
        mask4,
        scale,
    )


def visualize_ignores(ignore1, ignore2, ignore3, ignore4, scale=4):
    """
    False(ignore) = 흰색
    """
    return _visualize_binary_maps(
        ~ignore1,
        ~ignore2,
        ~ignore3,
        ~ignore4,
        scale,
    )


def draw_heads(img_tensor, head1, head2, head3, head4):
    """
    img_tensor : (3,H,W) torch.Tensor
    head       : (5,H,W) torch.Tensor
    """

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255.0 + 128.0).clip(0, 255).astype(np.uint8)

    # Head4부터 그린다.
    heads = [
        (head4.cpu().numpy(), 32, (0, 255, 0),   4),  # Green
        (head3.cpu().numpy(), 16, (0, 255, 255), 3),  # Yellow
        (head2.cpu().numpy(),  8, (255, 0, 0),   2),  # Blue
        (head1.cpu().numpy(),  4, (0, 0, 255),   1),  # Red
    ]

    out = img.copy()

    for head, stride, color, thickness in heads:

        heat = head[0]
        offset_x = head[1]
        offset_y = head[2]
        log_w = head[3]
        log_h = head[4]

        #ys, xs = np.where(heat > 0.5)
        batch_heatmaps = heat[np.newaxis, np.newaxis, ...]
        batch_peaks = get_final_multi_preds(batch_heatmaps, threshold=0.5, post_processing_method="simple")
        peaks = batch_peaks[0][0]
        if len(peaks) != 0:
            peaks = np.array(peaks)
            xs, ys = peaks[:, 0].astype(int), peaks[:, 1].astype(int)

            for iy, ix in zip(ys, xs):

                cx = (ix + offset_x[iy, ix]) * stride
                cy = (iy + offset_y[iy, ix]) * stride

                w = np.exp(log_w[iy, ix]) * stride
                h = np.exp(log_h[iy, ix]) * stride

                x1 = int(round(cx - w * 0.5))
                y1 = int(round(cy - h * 0.5))
                x2 = int(round(cx + w * 0.5))
                y2 = int(round(cy + h * 0.5))

                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    return out

def get_train_loader(conf):
    transform_img = transf_img.Compose([
        transf_img.PixelNoise(noise_factor=0.4),
        transf_img.Gamma(noise_factor=0.3),
        transf_img.Blur(blur=0.1)
    ])

    path_data = conf["DATASET"]["ROOT_TRAIN"]
    batch_size = conf["TRAIN"]["BATCH_SIZE"]
    shuffle = conf["TRAIN"]["SHUFFLE"]
    num_workers = conf["TRAIN"]["WORKERS"]

    dataset = WiderFaceCenterTrain(path_data, transform_img)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    return loader



if '__main__' == __name__:
    import yaml

    param_path = '../experiments/apag_widerface.yml'
    with open(param_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)


    train_loader = get_train_loader(conf)

    for batch in train_loader:

        imgs = batch["image"]
        heads = batch["heads"][0]
        masks = batch["masks"][0]
        ignores = batch["ignores"][0]

        print("img shape", imgs.shape)
        print("heads shape", heads.shape)
        print("masks shape", masks.shape)
        print("ignores shape", ignores.shape)

        # for b in range(len(imgs)):
        #
        #     img_check = draw_heads(imgs[b],batch["heads"][0][b],
        #                                     batch["heads"][1][b],
        #                                     batch["heads"][2][b],
        #                                     batch["heads"][3][b])
        #
        #     cv2.imshow('img_check', img_check)
        #     key = cv2.waitKey(0)
        #
        #     if key == ord('q'):
        #         break