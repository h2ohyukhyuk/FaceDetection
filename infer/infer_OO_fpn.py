
from net.OOFPNNet import Network
from utils.model_utils import load_model_from_path
from utils.decode_output import decode_heads, nms

import cv2
import numpy as np
import torch
import time
import cv2


def main():
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    model = Network()
    path_model = "/home/j/models/face_detection/OO-fpn-0000/159.pt"
    model = load_model_from_path(
        path_model,
        model,
        strict=False,
    )

    model = model.to(device)
    model.eval()

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/j/Downloads/tak2.mp4")

    colors = [
        (0, 255, 0),  # Head4 (stride32) - Green
        (0, 255, 255),  # Head3 (stride16) - Yellow
        (255, 0, 0),  # Head2 (stride8)  - Blue
        (0, 0, 255),  # Head1 (stride4)  - Red
    ]

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------------------------
        # Resize (384x256)
        # -------------------------------------------------
        img = cv2.resize(frame, (384, 256))

        img_input = (img.astype(np.float32) - 128.0) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, 0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        img_tensor = torch.from_numpy(img_input).float().to(device)

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------

        with torch.no_grad():

            p_head0 = model(img_tensor)

            p_head0[:, 0:1] = torch.sigmoid(p_head0[:, 0:1])

        heads = [
            (p_head0.cpu().numpy(), 4),
        ]

        rects_batch = decode_heads(heads, score_thresh=0.4)
        rects_nms = nms(rects_batch[0], iou_thresh=0.4)

        torch.cuda.synchronize()
        infer_time = (time.perf_counter() - t0) * 1000.0  # ms

        # -------------------------------------------------
        # Draw
        # -------------------------------------------------
        vis = img.copy()
        cv2.putText(
            vis,
            f"{infer_time:05.1f} ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        for rect in rects_nms:
            x1, y1, x2, y2, score = rect

            cv2.rectangle(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0,0,255),
                2)

            cv2.putText(
                vis,
                f"{score:.2f}",
                (int(x1), int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1,
            )

        # all_rects = []
        #
        # for (head, stride), color in zip(heads, colors):
        #
        #     rects = decode_heads([(head, stride)], score_thresh=0.5)[0]
        #
        #     for rect in rects:
        #
        #         keep = False
        #
        #         for rect_nms in rects_nms:
        #             if np.allclose(rect[:4], rect_nms[:4], atol=1e-3):
        #                 keep = True
        #                 break
        #
        #         if not keep:
        #             continue
        #
        #         x1, y1, x2, y2, score = rect
        #
        #         if score > 0.6:
        #             cv2.rectangle(
        #                 vis,
        #                 (int(x1), int(y1)),
        #                 (int(x2), int(y2)),
        #                 color,
        #                 2)
        #         else:
        #             cv2.rectangle(
        #                 vis,
        #                 (int(x1), int(y1)),
        #                 (int(x2), int(y2)),
        #                 (255,255,255),
        #                 2)
        #
        #         cv2.putText(
        #             vis,
        #             f"{score:.2f}",
        #             (int(x1), int(y1) - 4),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5,
        #             (255,255,255),
        #             1,
        #         )

        cv2.imshow("Face Detector", vis)

        key = cv2.waitKey(33) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()