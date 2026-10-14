"""Run the APAG FPN2 detector with FP16 weights and inputs on CUDA."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

# Allow execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from net.apag_fpn_net2 import Network
from utils.model_utils import load_model_from_path
from utils.decode_output import decode_heads, nms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--model', default='/home/j/models/face_detection/apag-fpn2-0000/180.pt')
    parser.add_argument('--model', default='/home/j/models/face_detection/apag-fpn2-fp16-qat-0000/001.pt')
    parser.add_argument('--source', default='0', help='Camera index or video path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    device = torch.device(args.device)
    if device.type != 'cuda':
        parser.error('--device must be a CUDA device, e.g. cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and CUDA-enabled PyTorch installation.')

    model = load_model_from_path(args.model, Network(), strict=False)
    model = model.to(device=device, dtype=torch.float16).eval()
    print(f'Device: {torch.cuda.get_device_name(device)}; model/input dtype: float16')
    # Fixed image size permits cuDNN to select efficient convolution kernels.
    torch.backends.cudnn.benchmark = True
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        warmup_input = torch.zeros(1, 3, 256, 384, device=device, dtype=torch.float16)
        for _ in range(args.warmup):
            model(warmup_input)
        torch.cuda.synchronize(device)

    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video source: {args.source}')
    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = cv2.resize(frame, (384, 256))
                # Preserve the original BGR normalization; round to FP16 afterwards.
                img_input = (img.astype(np.float32) - 128.0) / 255.0
                img_input = np.ascontiguousarray(img_input.transpose(2, 0, 1)[None])

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                img_tensor = torch.from_numpy(img_input).to(device=device, dtype=torch.float16)
                with torch.cuda.device(device):
                    start_event.record()
                    p_head0 = model(img_tensor)
                    end_event.record()
                # FP32 sigmoid, decoding and NMS avoid FP16 exp/coordinate overflow.
                head = p_head0.float()
                head[:, 0:1] = head[:, 0:1].sigmoid()
                heads = [(head.cpu().numpy(), 4)]
                rects_batch = decode_heads(heads, score_thresh=0.4)
                rects_nms = nms(rects_batch[0], iou_thresh=0.4)
                end_event.synchronize()
                forward_ms = start_event.elapsed_time(end_event)
                pipeline_ms = (time.perf_counter() - t0) * 1000.0

                vis = img.copy()
                cv2.putText(vis, f'FP16 GPU: {forward_ms:.1f} ms', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f'Input + infer + decode: {pipeline_ms:.1f} ms', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for x1, y1, x2, y2, score in rects_nms:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(vis, f'{score:.2f}', (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Face Detector FP16 CUDA', vis)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
