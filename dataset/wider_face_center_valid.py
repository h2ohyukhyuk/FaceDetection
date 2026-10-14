import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import cv2
import numpy as np
import random
from utils import transforms as transf_img
import json


class WiderFaceCenterValid(data.Dataset):
    def __init__(self, root):

        self.dst_width = 384
        self.dst_height = 256
        self.dst_rect_min = 10  # dst 크기 기준
        self.max_num_face = 30
        self.samples = []

        image_extensions = (".jpg", ".png")

        image_files = []

        for filename in os.listdir(root):
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, filename))

        image_files = sorted(image_files)
        
        for path_img in image_files:
            rects = []
            path_anno_json = path_img.replace("png", "json").replace("jpg", "json")
            if os.path.exists(path_anno_json):
                with open(path_anno_json, "r", encoding="utf-8") as f:
                    annos = json.load(f)

                    for anno in annos:
                        x, y, w, h = anno["box"]
                        rects.append([x, y, x+w, y+h])

            self.samples.append({"path_img":path_img, "rects":rects})


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):

        path_img = self.samples[index]["path_img"]
        img = cv2.imread(path_img)
        height, width, _ = img.shape
        scale_h = self.dst_height / height
        scale_w = self.dst_width / width
        img = cv2.resize(img, (int(self.dst_width), int(self.dst_height)))

        rects = self.samples[index]["rects"]
        rects = np.array(rects).astype(np.float32)
        rects_filter = np.zeros((0,4), dtype=np.float32)

        for rect in rects:
            x1, y1, x2, y2 = rect
            w, h = x2 - x1, y2 - y1

            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height or w <= 0 or h <= 0:
                #print("outside face rect", x1, y1, x2, y2, self.imgs_path[index])
                pass
            else:
                nx1, ny1, nx2, ny2 = int(x1*scale_w), int(y1*scale_h), int(x2*scale_w), int(y2*scale_h)

                size_max = max(nx2-nx1, ny2-ny1)
                if size_max > self.dst_rect_min:
                    rects_filter = np.append(rects_filter, np.array([nx1, ny1, nx2, ny2], dtype=np.float32).reshape(1, 4), axis=0)
                else:
                    img[ny1:ny2, nx1:nx2, :] = 128


        rects_t_f = np.zeros((self.max_num_face, 5), dtype=np.float32)
        for i, rt_t in enumerate(rects_filter):

            if i >= self.max_num_face:
                break
            x1, y1, x2, y2 = rt_t
            rects_t_f[i] = np.array([1, x1, y1, x2, y2], dtype=np.float32)

        # for rt in rects_t_f:
        #     ex, x1, y1, x2, y2 = rt.astype(int)
        #     if ex == 1:
        #         cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        #
        # cv2.imshow("",img)
        # cv2.waitKey(0)

        img = (img.astype(float) - 128.0) / 255.0
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
        rects_tensor = torch.from_numpy(rects_t_f).float()

        return {
            "image": img_tensor,
            "rects": rects_tensor
        }


def get_valid_loader(conf):


    root = '/home/j/database/face_detection_valid_aspa/'
    dataset = WiderFaceCenterValid(root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    return loader



if '__main__' == __name__:


    valid_loader = get_valid_loader("confg")

    for batch in valid_loader:

        imgs = batch["image"]
        rects = batch["rects"]
