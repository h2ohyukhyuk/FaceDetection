import numpy as np
import cv2
import random

from numpy.ma.core import reshape


class ImageBoxSRTF:

    def __init__(
        self,
        src_rect_min=24,
        dst_rect_min=14,
        dst_width=384,
        dst_height=256,
        rot=10,
        flip_prob=0.5,
        border_value=(128, 128, 128)
    ):
        self.src_rect_min = src_rect_min
        self.dst_width = dst_width
        self.dst_height = dst_height

        self.dst_rect_min = dst_rect_min
        self.dst_rect_max = max(self.dst_width, self.dst_height)
        self.rot = rot

        self.flip_prob = flip_prob
        self.border_value = border_value

    def get_params(self, src_rect):
        x1, y1, x2, y2 = src_rect
        w, h = x2 - x1, y2 - y1

        # scale
        src_rect_size = max(w, h)
        max_size = min(self.dst_rect_max, src_rect_size * 2.0)
        min_size = min(max_size-10,max(self.dst_rect_min, src_rect_size * 0.5))

        target_rect_size = np.exp(
            np.random.uniform(
                np.log(min_size),
                np.log(max_size)
            )
        )
        scale = target_rect_size / src_rect_size

        # translation
        margin = min( int(self.dst_height*0.45), max(5, int(target_rect_size*0.1)))
        dst_cx = random.randint(margin,self.dst_width - margin)
        dst_cy = random.randint(margin, self.dst_height - margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tx = dst_cx - cx
        ty = dst_cy - cy

        # rotation
        if random.random() < 0.5:
            rot = 0
        else:
            rot = random.uniform(-self.rot, self.rot)

        M = cv2.getRotationMatrix2D(
            (cx, cy),
            rot,
            scale
        )

        M[0, 2] += tx
        M[1, 2] += ty

        return M

    @staticmethod
    def transform_rects(M, rects):
        """
        Parameters
        ----------
        M : (2,3)
            cv2.getRotationMatrix2D() 등으로 생성한 Affine Matrix

        rects : (N,4)
            [x1, y1, x2, y2]

        Returns
        -------
        rects_out : (N,4)
            중심은 Affine 변환
            Width/Height는 Scale만 적용
            Rotation은 적용하지 않음
        """

        if len(rects) == 0:
            return np.asarray(rects, dtype=np.float32)

        rects = np.asarray(rects, dtype=np.float32)

        if rects.ndim != 2 or rects.shape[1] != 4:
            raise ValueError(
                f"rects shape must be (N,4), got {rects.shape}"
            )

        # 중심
        cx = (rects[:, 0] + rects[:, 2]) * 0.5
        cy = (rects[:, 1] + rects[:, 3]) * 0.5

        # 크기
        w = rects[:, 2] - rects[:, 0]
        h = rects[:, 3] - rects[:, 1]

        # Affine 적용
        centers = np.stack([cx, cy], axis=1)

        centers_h = np.concatenate(
            [centers, np.ones((len(rects), 1), dtype=np.float32)],
            axis=1
        )

        centers = centers_h @ M.T

        # Scale 추출
        scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)

        w *= scale
        h *= scale

        rects_out = np.empty_like(rects)

        rects_out[:, 0] = centers[:, 0] - w * 0.5
        rects_out[:, 1] = centers[:, 1] - h * 0.5
        rects_out[:, 2] = centers[:, 0] + w * 0.5
        rects_out[:, 3] = centers[:, 1] + h * 0.5

        return rects_out

    @staticmethod
    def flip_rects(rects, width):
        """
        rects : (N,4)
                [x1, y1, x2, y2]

        width : image width

        return
            flipped rects
        """

        rects = np.asarray(rects, dtype=np.float32)

        if rects.size == 0:
            return rects

        rects = rects.copy()

        x1 = rects[:, 0].copy()
        x2 = rects[:, 2].copy()

        rects[:, 0] = width - 1 - x2
        rects[:, 2] = width - 1 - x1

        return rects

    def random_crop(self, img):
        h, w = img.shape[:2]

        scale = random.uniform(0.2, 1.0)

        crop_w = int(w * scale)
        crop_h = int(h * scale)

        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

        crop = img[y:y + crop_h, x:x + crop_w]

        crop = cv2.resize(
            crop,
            (self.dst_width, self.dst_height),
            interpolation=cv2.INTER_LINEAR
        )

        return crop

    def __call__(self, img, rects):

        # no rect
        if len(rects) == 0:
            return self.random_crop(img), rects

        sizes = np.maximum(
            rects[:, 2] - rects[:, 0],
            rects[:, 3] - rects[:, 1]
        )

        valid_idx = np.where(sizes >= self.src_rect_min)[0]

        # no valid rect
        if len(valid_idx) == 0:
            for rt in rects:
                x1, y1, x2, y2 = rt.astype(int)
                img[y1:y2, x1:x2, :] = 128
            return self.random_crop(img), np.zeros((0,4), dtype=np.float32)
        else:
            idx = random.choice(valid_idx)
            src_rect = rects[idx]

            for i, rt in enumerate(rects):
                if i not in valid_idx:
                    x1, y1, x2, y2 = rt.astype(int)
                    img[y1:y2, x1:x2, :] = 128

            rects = rects[valid_idx]

        M = self.get_params(src_rect)

        img_t = cv2.warpAffine(
            img,
            M,
            (self.dst_width, self.dst_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

        rects_t = self.transform_rects(M, rects)

        # 랜덤 좌우 반전
        if random.random() < self.flip_prob:

            img_t = cv2.flip(img_t, 1)

            if len(rects_t):
                rects_t = self.flip_rects(rects_t, self.dst_width)

        return np.ascontiguousarray(img_t), rects_t


class ImageBoxBG:

    def __init__(
        self,
        dst_width=384,
        dst_height=256,
        rot=30,
        flip_prob=0.5,
        border_value=(128, 128, 128)

    ):
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.dst_size = max(dst_width, dst_height)

        self.dst_rect_max = max(self.dst_width, self.dst_height)
        self.rot = rot

        self.flip_prob = flip_prob
        self.border_value = border_value


    def get_params(self, src_rect):
        x1, y1, x2, y2 = src_rect
        w, h = x2 - x1, y2 - y1

        # scale
        src_rect_size = max(w, h)

        min_size = self.dst_size * 0.5
        max_size = self.dst_size * 2.0

        target_size = np.exp(
            np.random.uniform(
                np.log(min_size),
                np.log(max_size)
            )
        )
        scale = target_size / src_rect_size

        # translation
        margin = min( int(self.dst_height*0.45), max(5, int(target_size*0.1)))
        dst_cx = random.randint(margin,self.dst_width - margin)
        dst_cy = random.randint(margin, self.dst_height - margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tx = dst_cx - cx
        ty = dst_cy - cy

        # rotation
        rot = random.uniform(-self.rot, self.rot)

        M = cv2.getRotationMatrix2D(
            (cx, cy),
            rot,
            scale
        )

        M[0, 2] += tx
        M[1, 2] += ty

        return M


    def __call__(self, img, rects):

        for rt in rects:
            x1, y1, x2, y2 = rt.astype(int)
            img[y1:y2, x1:x2, :] = 128

        h, w = img.shape[:2]

        scale = random.uniform(0.2, 0.95)

        crop_w = int(w * scale)
        crop_h = int(h * scale)

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        x2, y2 = x1 + crop_w, y1 + crop_h
        src_rect = [x1, y1, x2, y2]

        M = self.get_params(src_rect)

        img_t = cv2.warpAffine(
            img,
            M,
            (self.dst_width, self.dst_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )


        # 랜덤 좌우 반전
        if random.random() < self.flip_prob:

            img_t = cv2.flip(img_t, 1)

        return np.ascontiguousarray(img_t), np.zeros((0, 4), dtype=np.float32)