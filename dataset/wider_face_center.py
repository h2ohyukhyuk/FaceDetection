from dataset.wider_face_center_train import WiderFaceCenterTrain
from dataset.wider_face_center_valid import WiderFaceCenterValid
from torch.utils.data import DataLoader
from utils import transforms as transf_img
import numpy as np
from utils.heat_map import get_final_multi_preds

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def get_valid_loader(conf):


    path_root = conf["DATASET"]["ROOT_VALID"]
    batch_size = conf["VALID"]["BATCH_SIZE"]
    shuffle = conf["VALID"]["SHUFFLE"]
    num_workers = conf["VALID"]["WORKERS"]

    dataset = WiderFaceCenterValid(path_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader



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

if '__main__' == __name__:
    import yaml
    import cv2

    param_path = '../experiments/apag_widerface.yml'
    with open(param_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)


    if False:
        train_loader = get_train_loader(conf)
        for i, batch in enumerate(train_loader):

            imgs = batch["image"]
            heads = batch["heads"][0]
            masks = batch["masks"][0]
            ignores = batch["ignores"][0]

            print("img shape", imgs.shape)
            print("heads shape", heads.shape)
            print("masks shape", masks.shape)
            print("ignores shape", ignores.shape)

            for b in range(len(imgs)):

                img_check = draw_heads(imgs[b], batch["heads"][0][b],
                                       batch["heads"][1][b],
                                       batch["heads"][2][b],
                                       batch["heads"][3][b])

                cv2.imshow('img_check', img_check)
                key = cv2.waitKey(0)

                if key == ord('q'):
                    break

    if True:
        valid_loader = get_valid_loader(conf)
        for batch in valid_loader:

            imgs = batch["image"]
            rectss = batch["rects"]

            print("img shape", imgs.shape)
            print("rectss shape", rectss.shape)

            for b in range(len(imgs)):
                img = imgs[b].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255.0 + 128.0).clip(0, 255).astype(np.uint8)
                img = np.ascontiguousarray(img)
                rects = rectss[b].cpu().numpy()

                for rect in rects:
                    e, x1, y1, x2, y2 = rect.astype(int)

                    if e != 1:
                        break

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.imshow("",img)
                cv2.waitKey(0)
