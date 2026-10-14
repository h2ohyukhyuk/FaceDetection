import numpy as np
from utils.heat_map import get_final_multi_preds


def decode_heads(heads, score_thresh=0.5):
    """
    heads : [
        (head4, stride32),
        (head3, stride16),
        (head2, stride8),
        (head1, stride4),
    ]

    head : (B,5,H,W) numpy

    return
        rects_batch : list of length B
            rects_batch[b] = (N,5)
            [x1,y1,x2,y2,score]
    """

    batch_size = heads[0][0].shape[0]

    rects_batch = []

    for b in range(batch_size):

        rects = []

        for head, stride in heads:

            heat = head[b, 0]
            offx = head[b, 1]
            offy = head[b, 2]
            logw = head[b, 3]
            logh = head[b, 4]

            # ys, xs = np.where(heat > score_thresh)
            batch_heatmaps = heat[np.newaxis, np.newaxis, ...]
            batch_peaks = get_final_multi_preds(batch_heatmaps, threshold=score_thresh, min_dist=2, post_processing_method="simple")
            peaks = batch_peaks[0][0]

            if len(peaks) != 0:
                peaks = np.array(peaks)
                xs, ys = peaks[:, 0].astype(int), peaks[:, 1].astype(int)

                for iy, ix in zip(ys, xs):

                    score = heat[iy, ix]

                    cx = (ix + offx[iy, ix]) * stride
                    cy = (iy + offy[iy, ix]) * stride

                    w = np.exp(logw[iy, ix]) * stride
                    h = np.exp(logh[iy, ix]) * stride

                    rects.append([
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                        score,
                    ])

        if len(rects):
            rects = np.asarray(rects, dtype=np.float32)
        else:
            rects = np.empty((0, 5), dtype=np.float32)

        rects_batch.append(rects)

    return rects_batch

def nms(rects, iou_thresh=0.5):
    """
    rects : (N,5)
        [x1, y1, x2, y2, score]
    """

    if len(rects) == 0:
        return rects

    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    score = rects[:, 4]

    area = (x2 - x1) * (y2 - y1)

    order = score.argsort()[::-1]

    keep = []

    while order.size > 0:

        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h

        iou = inter / (area[i] + area[order[1:]] - inter + 1e-6)

        inds = np.where(iou < iou_thresh)[0]

        order = order[inds + 1]

    return rects[keep]


def nms(rects, iou_thresh=0.5):
    """
    rects : (N,5)
        [x1, y1, x2, y2, score]
    """

    if len(rects) == 0:
        return rects

    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    score = rects[:, 4]

    area = (x2 - x1) * (y2 - y1)

    order = score.argsort()[::-1]

    keep = []

    while order.size > 0:

        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h

        iou = inter / (area[i] + area[order[1:]] - inter + 1e-6)

        inds = np.where(iou < iou_thresh)[0]

        order = order[inds + 1]

    return rects[keep]


def calc_metric(gt_rects, pred_rects, iou_thresh=0.5):
    """
    gt_rects   : (N,4)
    pred_rects : (M,5)
    """

    matched = np.zeros(len(gt_rects), dtype=bool)

    tp = 0
    fp = 0

    order = np.argsort(pred_rects[:,4])[::-1]

    for idx in order:

        pred = pred_rects[idx]

        best_iou = 0.
        best_gt = -1

        px1, py1, px2, py2 = pred[:4]

        for i, gt in enumerate(gt_rects):

            if matched[i]:
                continue

            gx1, gy1, gx2, gy2 = gt

            xx1 = max(px1, gx1)
            yy1 = max(py1, gy1)
            xx2 = min(px2, gx2)
            yy2 = min(py2, gy2)

            w = max(0., xx2-xx1)
            h = max(0., yy2-yy1)

            inter = w*h

            area_p = (px2-px1)*(py2-py1)
            area_g = (gx2-gx1)*(gy2-gy1)

            iou = inter/(area_p+area_g-inter+1e-6)

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_thresh:
            matched[best_gt] = True
            tp += 1
        else:
            fp += 1

    fn = len(gt_rects)-tp

    return tp, fp, fn