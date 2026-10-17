"""Camera/video/image inference for APAG FPN2 with five facial landmarks."""
import argparse
from pathlib import Path
import sys,time
import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from net.apag_fpn_net2_landmark import Network
from utils.decode_landmarks import decode_landmarks

model_path = "/home/j/models/face_detection/apag-fpn2-wider-original-0000-landmark-heatmap/200.pt"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=model_path, help='Checkpoint trained with the landmark model')
    p.add_argument('--source',default='0',help='Camera index, video or image file')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fp16',action='store_true',help='Use FP16 model/input on CUDA')
    p.add_argument('--score',type=float,default=.4)
    p.add_argument('--landmark-score',type=float,default=.1,help='Minimum point confidence; missing points are not drawn')
    p.add_argument('--iou',type=float,default=.4)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--no-display',action='store_true')
    p.add_argument('--output',help='Save annotated image or .mp4/.avi video')
    p.add_argument('--max-frames',type=int,default=0,help='0: until end or q/Esc')
    a=p.parse_args();device=torch.device(a.device)
    if a.fp16 and device.type!='cuda':p.error('--fp16 requires CUDA')
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; check the NVIDIA driver')
    if not (0<=a.score<=1 and 0<=a.iou<=1 and 0<=a.landmark_score<=1) or a.warmup<0 or a.max_frames<0:p.error('Invalid threshold/frame/warmup argument')
    dtype=torch.float16 if a.fp16 else torch.float32
    model=Network()
    checkpoint=torch.load(a.model,map_location='cpu',weights_only=False)
    state=checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    # Detection-only checkpoints must not silently leave landmarks untrained.
    model.load_state_dict(state,strict=True)
    model=model.to(device=device,dtype=dtype).eval()
    print(f'Device={device} dtype={dtype}; landmark order: left eye, right eye, nose, left mouth, right mouth (image coordinates)')
    is_image=Path(a.source).suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')
    cap=None;writer=None;count=0;totals=np.zeros(3)
    if a.output:Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if is_image:
        still=cv2.imread(a.source)
        if still is None:raise RuntimeError(f'Cannot read image: {a.source}')
    else:
        cap=cv2.VideoCapture(int(a.source) if a.source.isdecimal() else a.source)
        if not cap.isOpened():cap.release();raise RuntimeError(f'Cannot open source: {a.source}')
    colors=[(0,255,255),(255,255,0),(0,255,0),(255,0,255),(0,128,255)]
    try:
        with torch.inference_mode():
            for _ in range(a.warmup):model(torch.zeros(1,3,256,384,device=device,dtype=dtype))
            if device.type=='cuda':torch.cuda.synchronize(device)
            while True:
                if is_image:frame=still.copy()
                else:
                    ok,frame=cap.read()
                    if not ok:break
                t0=time.perf_counter()
                img=cv2.resize(frame,(384,256))
                array=np.ascontiguousarray(((img.astype(np.float32)-128)/255).transpose(2,0,1)[None])
                tensor=torch.from_numpy(array).to(device=device,dtype=dtype)
                if device.type=='cuda':torch.cuda.synchronize(device)
                t1=time.perf_counter();raw=model(tensor)
                if device.type=='cuda':torch.cuda.synchronize(device)
                t2=time.perf_counter()
                rects=decode_landmarks(raw,score_thresh=a.score,iou_thresh=a.iou,landmark_score_thresh=a.landmark_score)[0]
                t3=time.perf_counter()
                timings=np.array([(t2-t1)*1000,(t3-t2)*1000,(t3-t0)*1000])
                totals+=timings;count+=1
                # Draw on original image, restoring its aspect ratio and coordinates.
                sy,sx=frame.shape[0]/256,frame.shape[1]/384
                for row in rects:
                    x1,y1,x2,y2,score=row[:5]
                    cv2.rectangle(frame,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),(0,0,255),2)
                    cv2.putText(frame,f'{score:.2f}',(round(x1*sx),round(y1*sy)-4),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    for (x,y),color in zip(row[5:].reshape(5,2),colors):
                        if not np.isfinite([x,y]).all():continue
                        cv2.circle(frame,(round(x*sx),round(y*sy)),3,color,-1)
                cv2.putText(frame,f'forward {timings[0]:.1f} ms | post {timings[1]:.1f} | total {timings[2]:.1f}',
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
                if a.output:
                    if is_image:
                        if not cv2.imwrite(a.output,frame):raise RuntimeError('Image save failed')
                    else:
                        if writer is None:
                            fps=cap.get(cv2.CAP_PROP_FPS)
                            if not np.isfinite(fps) or fps<=0:fps=25
                            codec='MJPG' if Path(a.output).suffix.lower()=='.avi' else 'mp4v'
                            writer=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*codec),fps,(frame.shape[1],frame.shape[0]))
                            if not writer.isOpened():raise RuntimeError('Video writer failed to open')
                        writer.write(frame)
                if not a.no_display:
                    cv2.imshow('APAG FPN2 landmarks',frame)
                    if cv2.waitKey(0 if is_image else 1)&0xFF in (27,ord('q')):break
                if is_image or (a.max_frames and count>=a.max_frames):break
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        if not a.no_display:cv2.destroyAllWindows()
    if count:
        avg=totals/count
        print(f'{count} frames | mean forward={avg[0]:.3f} ms post={avg[1]:.3f} ms total={avg[2]:.3f} ms (excludes capture/drawing/display)')

if __name__=='__main__':main()
