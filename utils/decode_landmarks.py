"""Raw Bx20xHxW -> [xyxy, score, 5*(x,y)] in input pixels.

Each point is the strongest heatmap location inside the detected face, refined
by that point's local offset. Points below threshold are returned as NaN pairs.
"""
import numpy as np
import torch
from utils.heat_map import extract_peaks
from utils.decode_output import nms


def decode_landmarks(output, stride=4, score_thresh=.4, iou_thresh=.4,
                     landmark_score_thresh=.1):
    if isinstance(output, torch.Tensor):
        output = output.detach().float().cpu().numpy()
    if output.ndim != 4 or output.shape[1] != 20:
        raise ValueError('Expected raw Bx20xHxW heatmap/offset output; legacy landmark checkpoints require retraining')
    batches = []
    for head in output:
        heat = 1 / (1 + np.exp(-np.clip(head[0], -80, 80)))
        peaks = extract_peaks(heat, threshold=score_thresh, min_dist=2)
        if not len(peaks):
            batches.append(np.empty((0, 15), np.float32))
            continue
        xs, ys = peaks[:, 0].astype(int), peaks[:, 1].astype(int)
        cx = (xs + head[1, ys, xs]) * stride
        cy = (ys + head[2, ys, xs]) * stride
        with np.errstate(over='ignore', invalid='ignore'):
            w = np.exp(head[3, ys, xs]) * stride
            h = np.exp(head[4, ys, xs]) * stride
        rows = np.column_stack((cx-w/2, cy-h/2, cx+w/2, cy+h/2, heat[ys, xs])).astype(np.float32)
        valid = np.isfinite(rows).all(axis=1) & (w > 0) & (h > 0)
        rows = nms(rows[valid], iou_thresh=iou_thresh)
        result = np.full((len(rows), 15), np.nan, np.float32)
        result[:, :5] = rows[:, :5]
        height, width = head.shape[1:]
        landmark_heat = 1 / (1 + np.exp(-np.clip(head[5:10], -80, 80)))
        for i, row in enumerate(rows):
            # Include cells whose local offsets can reach across a box edge.
            x1, y1 = np.clip(np.floor(row[:2]/stride), [0, 0], [width, height]).astype(int)
            x2, y2 = np.clip(np.ceil(row[2:4]/stride), [0, 0], [width, height]).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            yy, xx = np.mgrid[y1:y2, x1:x2]
            for k in range(5):
                px = (xx + head[10+2*k, y1:y2, x1:x2]) * stride
                py = (yy + head[11+2*k, y1:y2, x1:x2]) * stride
                scores = landmark_heat[k, y1:y2, x1:x2]
                inside = (np.isfinite(px) & np.isfinite(py) & np.isfinite(scores) &
                          (px >= row[0]) & (px <= row[2]) & (py >= row[1]) & (py <= row[3]) &
                          (scores >= landmark_score_thresh))
                if not inside.any():
                    continue
                index = np.argmax(np.where(inside, scores, -np.inf))
                result[i, 5+2*k:7+2*k] = (px.flat[index], py.flat[index])
        batches.append(result)
    return batches
