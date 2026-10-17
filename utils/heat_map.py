import numpy as np
import math
import cv2
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

import matplotlib.pyplot as plt

def draw_heatmap(heatmap, c_x:float, c_y:float, stride:int, sigma:float):
    '''
    genelly size of heatmap is down-sampled by factor stride from size of input image
    :param heatmap: gaussian heatmap will be drawed, is resized by stride
    :param c_x: position x of input image
    :param c_y: position y of input image
    :param stride: interger, sampling rate of input image
    :param sigma: sigma of heatmap of original image
    :return:
    '''

    size_y, size_x = heatmap.shape[0:2]
    # stride pixel maps to 1 pixel
    interp_start = stride/2.0 - 0.5 #0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    multiplier = 2.0 * sigma * sigma

    for row in range(size_y):
        y = row*stride + interp_start
        y_menos_center_pt_sqr = (c_y-y)*(c_y-y)
        for col in range(size_x):
            x = col*stride + interp_start
            d2 = (c_x-x)*(c_x-x) + y_menos_center_pt_sqr
            exponent = d2 / multiplier

            if exponent <= 4.6052:
                value = heatmap[row, col]
                heatmap[row, col] = min(1.0, max(value, math.exp(-exponent)))

def draw_heatmap_ignore_mask(mask, c_x:float, c_y:float, stride:int, sigma:float):
    '''
    genelly size of heatmap is down-sampled by factor stride from size of input image
    :param heatmap: gaussian heatmap will be drawed, is resized by stride
    :param mask: bool heatmap mask
    :param c_x: position x of input image
    :param c_y: position y of input image
    :param stride: interger, sampling rate of input image
    :param sigma: sigma of heatmap of original image
    :return:
    '''

    size_y, size_x = mask.shape[0:2]
    # stride pixel maps to 1 pixel
    interp_start = stride/2.0 - 0.5 #0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    multiplier = 2.0 * sigma * sigma

    for row in range(size_y):
        y = row*stride + interp_start
        y_menos_center_pt_sqr = (c_y-y)*(c_y-y)
        for col in range(size_x):
            x = col*stride + interp_start
            d2 = (c_x-x)*(c_x-x) + y_menos_center_pt_sqr
            exponent = d2 / multiplier

            if exponent <= 1.8:
                mask[row, col] = False

# https://github.com/NiteshBharadwaj/part-affinity/blob/master/src/evaluation/post.py
def find_peaks(heatmap, threshod:float):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (dataset2d array) where we want to find peaks
    :return: dataset2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """
    peaks_binary = (maximum_filter(heatmap, footprint=generate_binary_structure(
        2, 1)) == heatmap) * (heatmap > threshod)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

# https://github.com/NiteshBharadwaj/part-affinity/blob/master/src/evaluation/post.py
def compute_resized_coords(coords, resizeFactor):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5

# https://github.com/NiteshBharadwaj/part-affinity/blob/master/src/evaluation/post.py
def NMS(heatmap, threshod:float, upsampFactor:float=1.0, refine_center:bool=True, gaussian_filt:bool=False):
    """
    NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
    :param heatmaps: set of grayscale images on which to find local maxima (dataset3d np.array,
    with dimensions image_height x image_width x num_heatmaps)
    :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
    Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
    :param bool_refine_center: Flag indicating whether:
     - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
     - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
     fine-tune the location of the peak at the resolution of the original input image
    :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
    to each upsampled patch before fine-tuning the location of each peak.
    :return: a NUM_JOINTS x 4 np.array where each row represents a joint type (0=nose, 1=neck...)
    and the columns indicate the {x,y} position, the score (probability) and a unique id (counter)
    """
    # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to heatmap_avg and
    # then performing NMS to find peaks, this step can be sped up by ~25-50x by:
    # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs 250-280ms on RoG
    # 1. Perform NMS at (low-res) CPM's output resolution
    # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
    # 2. Once a peak is found, take a patch of 5x5 centered around the peak, upsample it, and
    # fine-tune the position of the actual maximum.
    #  '-> That's equivalent to having found the peak on heatmap_avg, but much faster because we only
    #      upsample and scan the 5x5 patch instead of the full (e.g.) 480x640
    peak_coords = find_peaks(heatmap, threshod=threshod)
    #print('peak_coords: ', peak_coords)
    peaks = np.zeros(shape=(len(peak_coords), 3))

    win_size = 2
    for i, peak in enumerate(peak_coords):
        if refine_center:
            # Take a small patch around each peak and only upsample that
            # tiny region
            x_min, y_min = np.maximum( 0, peak - win_size)
            x_max, y_max = np.minimum( np.array(heatmap.T.shape) - 1, peak + win_size)
            patch = heatmap[y_min:y_max + 1, x_min:x_max + 1]

            map_upsamp = cv2.resize( patch, None, None, upsampFactor, upsampFactor, cv2.INTER_CUBIC)

            # Gaussian filtering takes an average of 0.8ms/peak (and there might be
            # more than one peak per joint!) -> For now, skip it (it's
            # accurate enough)
            map_upsamp = gaussian_filter( map_upsamp, 3) if gaussian_filt else map_upsamp

            # Obtain the coordinates of the maximum value in the patch
            location_of_max = np.unravel_index( map_upsamp.argmax(), map_upsamp.shape)

            # Remember that peaks indicates [x,y] -> need to reverse it for [y,x]
            location_of_patch_center = compute_resized_coords( peak[::-1] - [y_min, x_min], upsampFactor)

            # Calculate the offset wrt to the patch center where the actual
            # maximum is
            refined_center = (location_of_max - location_of_patch_center)[::-1] # [y,x] -> [x,y]
            peak_score = map_upsamp[location_of_max]
        else:
            refined_center = [0, 0]
            # Flip peak coordinates since they are [x,y] instead of [y,x]
            peak_score = heatmap[tuple(peak[::-1])]

        upsamp_peak = compute_resized_coords(peak, upsampFactor) + refined_center

        peaks[i, :] = (int(round(upsamp_peak[0])), int(round(upsamp_peak[1])), peak_score)
        #print(peaks[i, :] )

    return peaks

def visualize_heatmap(img, heatmap, size=None, interp=cv2.INTER_CUBIC, normalize_max=False):

    if heatmap is None:
        return img

    if len(heatmap.shape) == 4:
        heatmap = heatmap[0]

    heatmap = heatmap.max(axis=0)

    if normalize_max:
        heatmap = (heatmap / (heatmap.max() + 1e-8) * 254.).astype('uint8')
    else:
        heatmap = (np.clip(heatmap, 0.0, 1.0)* 255.).astype('uint8')

    if size is not None:
        heatmap = cv2.resize(heatmap, dsize=size, interpolation=interp)

    heatmap = cv2.applyColorMap(heatmap, colormap=cv2.COLORMAP_JET)
    heatmap = heatmap*0.5 + img*0.5
    heatmap = np.array(heatmap,dtype=np.uint8)

    return heatmap

def visualize_mask(img, heat_maps, displayname = ''):
    heat_maps = heat_maps.min(axis=0)
    heat_maps = (heat_maps/(heat_maps.max() + 1e-8) * 254.).astype('uint8')
    img = img.copy()
    colored = cv2.applyColorMap(heat_maps, cv2.COLORMAP_JET)

    img_h, img_w = img.shape[0:2]
    colored_h, colored_w = colored.shape[0:2]

    if img_h != colored_h or img_w != colored_w:
        colored = cv2.resize(colored, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)

    if displayname != '':
        cv2.imshow(displayname, img)
        cv2.waitKey()

    return img

class HeatMap():
    def __init__(self, input_h, input_w, stride, sigma):
        self.input_w = input_w
        self.input_h = input_h
        self.output_h = input_h // stride
        self.output_w = input_w // stride
        self.stride = stride
        self.interp_start = stride / 2.0 - 0.5
        self.multiplier = 2.0 * sigma * sigma
        self.gaussian_size = int(6.0 * sigma)

        self.output_gaussian_size = self.gaussian_size // stride
        self.output_gaussian_size_half = self.output_gaussian_size // 2

        self.xs = np.arange(0, self.output_w, 1, np.float32) * self.stride + self.interp_start
        self.ys = np.arange(0, self.output_h, 1, np.float32) * self.stride + self.interp_start

    def get_gaussian(self, c_x:float, c_y:float):
        sc_x, sc_y  = math.floor(c_x/self.stride), math.floor(c_y/self.stride)

        st_x, end_x = sc_x - self.output_gaussian_size_half, sc_x + self.output_gaussian_size_half
        st_y, end_y = sc_y - self.output_gaussian_size_half, sc_y + self.output_gaussian_size_half

        st_x, end_x = max(0, st_x), min(self.output_w, end_x)
        st_y, end_y = max(0, st_y), min(self.output_h, end_y)

        xs_roi = self.xs[st_x:end_x]
        ys_roi = self.ys[st_y:end_y][:, np.newaxis]

        gaussian = np.exp(- ((xs_roi - c_x) ** 2 + (ys_roi - c_y) ** 2) / (self.multiplier))

        return st_x, end_x, st_y, end_y, gaussian

    def __call__(self, heatmap, c_x:float, c_y:float):

        if heatmap.shape[0] != self.output_h or heatmap.shape[1] != self.output_w:
            raise ValueError('output heatmap size is different', heatmap.shape, self.output_h, self.output_w)

        st_x, end_x, st_y, end_y, gaussian = self.get_gaussian(c_x, c_y)
        heatmap[st_y:end_y, st_x:end_x] = np.maximum(heatmap[st_y:end_y, st_x:end_x], gaussian)

        return heatmap


class HeatMapIgnore():
    def __init__(self, input_size, stride, sigma):
        self.hm_drawer = HeatMap(input_size, stride, sigma)

    def __call__(self, heatmap_ignore_mask, c_x:float, c_y:float):

        st_x, end_x, st_y, end_y, gaussian = self.hm_drawer.get_gaussian(c_x, c_y)

        mask_ignore = gaussian > 0.1652

        heatmap_ignore_mask_roi = heatmap_ignore_mask[st_y:end_y, st_x:end_x]
        heatmap_ignore_mask_roi[mask_ignore] = False
        heatmap_ignore_mask[st_y:end_y, st_x:end_x] = heatmap_ignore_mask_roi

        return heatmap_ignore_mask


def extract_peaks(heatmap, threshold=0.1, min_dist=9):
    """Score-ordered Euclidean suppression, matching the original greedy rule."""
    ys, xs = np.where(heatmap > threshold)
    scores = heatmap[ys, xs]
    order = np.argsort(scores)[::-1]  # Preserve original equal-score ordering.
    blocked = np.zeros(heatmap.shape, dtype=bool)
    radius = max(0, int(np.ceil(abs(min_dist))) - 1)
    dy, dx = np.mgrid[-radius:radius+1, -radius:radius+1]
    disk = dx * dx + dy * dy < min_dist ** 2
    height, width = heatmap.shape
    peaks = []
    for idx in order:
        x, y = xs[idx], ys[idx]
        if blocked[y, x]:
            continue
        peaks.append((x, y, scores[idx]))
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        blocked[y0:y1, x0:x1] |= disk[y0-y+radius:y1-y+radius, x0-x+radius:x1-x+radius]
    return np.array(peaks, dtype=np.float32)


def gradient_refine(heatmap, peaks, offset=0.125):
    """
    Gradient sign based keypoint refinement

    Args:
        heatmap : (H, W) float ndarray
        peaks   : (N, 2) -> (x, y)
        offset  : sub-pixel offset

    Returns:
        refined_peaks : (N, 2) float32
    """

    H, W = heatmap.shape

    peaks = np.asarray(peaks, dtype=np.float32)

    if peaks.size == 0:
        return peaks.copy()

    refined = peaks.copy()

    for i, peak in enumerate(peaks):

        px = int(math.floor(peak[0] + 0.5))
        py = int(math.floor(peak[1] + 0.5))

        if 1 < px < W - 1 and 1 < py < H - 1:

            diff = np.array(
                [
                    heatmap[py, px + 1] - heatmap[py, px - 1],
                    heatmap[py + 1, px] - heatmap[py - 1, px],
                ],
                dtype=np.float32,
            )

            refined[i] += np.sign(diff) * offset

    return refined

def dark_refine(heatmap, peaks, gaussian_blur=7):
    """
    DARK keypoint refinement

    Args:
        heatmap : (H, W) float32 ndarray
        peaks   : [(x,y), ...] 또는 (N,2) ndarray
        gaussian_blur : Gaussian blur kernel size

    Returns:
        refined_peaks : (N,2) float32 ndarray
    """

    H, W = heatmap.shape

    hm = heatmap.astype(np.float32)

    if gaussian_blur > 0:
        hm = cv2.GaussianBlur(
            hm,
            (gaussian_blur, gaussian_blur),
            0
        )

    hm = np.maximum(hm, 1e-10)
    hm = np.log(hm)

    refined = []

    for peak in peaks:

        x, y = peak

        x0 = int(round(x))
        y0 = int(round(y))

        # 경계점은 보정하지 않음
        if x0 < 1 or x0 >= W - 1 or y0 < 1 or y0 >= H - 1:
            refined.append([float(x), float(y)])
            continue

        # gradient
        dx = 0.5 * (hm[y0, x0 + 1] - hm[y0, x0 - 1])
        dy = 0.5 * (hm[y0 + 1, x0] - hm[y0 - 1, x0])

        grad = np.array([dx, dy], dtype=np.float32)

        # Hessian
        dxx = hm[y0, x0 + 1] - 2 * hm[y0, x0] + hm[y0, x0 - 1]
        dyy = hm[y0 + 1, x0] - 2 * hm[y0, x0] + hm[y0 - 1, x0]

        dxy = 0.25 * (
            hm[y0 + 1, x0 + 1]
            - hm[y0 + 1, x0 - 1]
            - hm[y0 - 1, x0 + 1]
            + hm[y0 - 1, x0 - 1]
        )

        hessian = np.array(
            [
                [dxx, dxy],
                [dxy, dyy]
            ],
            dtype=np.float32
        )

        try:
            offset = -np.linalg.solve(hessian, grad)

            # DARK 구현들에서 흔히 넣는 안전장치
            if np.all(np.abs(offset) <= 1.0):
                x += offset[0]
                y += offset[1]

        except np.linalg.LinAlgError:
            pass

        refined.append([float(x), float(y)])

    return np.asarray(refined, dtype=np.float32)

def get_final_multi_preds(batch_heatmaps, threshold=0.5, min_dist=2, post_processing_method='DARK'):
    """
    params: batch_heatmaps  [b, num_joints, h, w]   np.array    np.float
    params: post_processing_method  str "method of post-processing 'simple', 'grad', 'DARK'"
    params: center
    params: scale
    """
    if post_processing_method not in ['simple', 'grad', 'DARK']:
        raise ValueError('post_processing_method should be one of simple, grad, DARK : %s' % (post_processing_method))


    batch_peaks = []
    for heatmaps in batch_heatmaps:
        list_peaks = []
        for heatmap in heatmaps:

            #peaks = NMS(heatmap,threshold=threshold, refine_center=False)

            peaks = extract_peaks(heatmap, threshold, min_dist=min_dist)

            if len(peaks) > 0:
                if post_processing_method == "grad":
                    peaks[:,0:2] = gradient_refine(heatmap, peaks[:,0:2])
                elif post_processing_method == "DARK":
                    peaks[:,0:2] = dark_refine(heatmap, peaks[:,0:2], gaussian_blur=7)

            list_peaks.append(peaks)
        batch_peaks.append(list_peaks)

    return batch_peaks


if __name__ == '__main__':

    c_x,  c_y =100.2, 127.51
    print('center: ', c_x,  c_y)
    stride = 16
    w, h = 224, 224
    sigma = 11.0
    resize_w, resize_h = w//stride, h//stride
    heatmap = np.zeros(shape=(h, w), dtype=np.float)
    heatmap_stride2 = np.zeros(shape=(resize_h, resize_w), dtype=np.float)

    draw_heatmap(heatmap, c_x=c_x, c_y=c_y, stride=1, sigma=sigma)
    draw_heatmap(heatmap_stride2, c_x=c_x, c_y=c_y, stride=stride, sigma=sigma )
    resize_heatmap = cv2.resize(heatmap, dsize=(resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

    # img_hm_st = np.array( heatmap_stride2*255.0, dtype=np.uint8)
    # cv2.imwrite('E:/hm.png', img_hm_st)
    # cv2.imshow('', img_hm_st)
    # cv2.waitKey()

    print(heatmap_stride2.max())
    peaks = NMS(heatmap_stride2, threshod=0.1, upsampFactor=stride)
    print(peaks)


    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(heatmap)
    axs[0,1].imshow(heatmap_stride2)
    axs[1,0].imshow(heatmap_stride2)
    axs[1,1].imshow(resize_heatmap)
    plt.show()