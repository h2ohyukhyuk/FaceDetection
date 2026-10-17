"""Heavy-blur removal must erase pixels and all targets before augmentation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from dataset.wider_face_landmark2 import LandmarkTrain


class WiderBlurFilterTests(unittest.TestCase):
    def make_dataset(self, root, only_blur=False):
        image_dir=root/'WIDER_train/images/event'
        image_dir.mkdir(parents=True)
        cv2.imwrite(str(image_dir/'test.jpg'),np.full((256,384,3),200,np.uint8))
        # blur=0 with heavy occlusion, blur=1, blur=2.
        boxes=[[20,20,32,32,0,0,0,0,2,0],
               [100,20,32,32,1,0,0,0,0,0],
               [180,20,32,32,2,0,0,0,0,0]]
        if only_blur:boxes=boxes[2:]
        split=root/'wider_face_split';split.mkdir()
        (split/'wider_face_train_bbx_gt.txt').write_text(
            'event/test.jpg\n'+str(len(boxes))+'\n'+''.join(' '.join(map(str,b))+'\n' for b in boxes))
        rows=[]
        for b in boxes:
            x,y,w,h=b[:4]
            points=[x+8,y+8,0,x+24,y+8,0,x+16,y+16,0,x+8,y+24,0,x+24,y+24,0]
            rows.append(' '.join(map(str,b[:4]+points+[1])))
        labels=split/'label.txt';labels.write_text('# event/test.jpg\n'+'\n'.join(rows)+'\n')
        dataset=LandmarkTrain({'DATASET':{'ROOT_TRAIN':str(root),'LANDMARK_LABEL':str(labels)},
                               'MODEL':{'INPUT_SIZE':[256,384]}},lambda x:x)
        dataset.train_bg_prob=0
        dataset.transform_srtf.get_params=lambda box:np.array([[1,0,0],[0,1,0]],np.float32)
        dataset.transform_srtf.random_crop=lambda image:image
        return dataset

    def test_only_heavy_blur_removed_from_pixels_boxes_and_landmarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=self.make_dataset(Path(tmp))
            self.assertEqual(len(dataset.samples[0][1]),2)
            self.assertEqual(len(dataset.blur_boxes[0]),1)
            with patch('dataset.wider_face_landmark2.random.random',return_value=1):
                batch=dataset[0]
            image=batch['image'].numpy()
            self.assertTrue((image[:,20:52,180:212]==0).all())
            self.assertTrue((image[:,20:52,20:52]>0).all())
            self.assertTrue((image[:,20:52,100:132]>0).all())
            self.assertEqual(batch['heads'][0][0,9,9].item(),1)
            self.assertEqual(batch['heads'][0][0,9,29].item(),1)
            self.assertEqual(batch['heads'][0][:,5:13,45:53].count_nonzero().item(),0)
            self.assertEqual(batch['landmark_heatmaps'][:,:,45:53].count_nonzero().item(),0)
            self.assertFalse(batch['landmark_mask'][:,:,45:53].any())

    def test_image_with_only_heavy_blur_becomes_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=self.make_dataset(Path(tmp),only_blur=True)
            with patch('dataset.wider_face_landmark2.random.random',return_value=1):
                batch=dataset[0]
            self.assertTrue((batch['image'][:,20:52,180:212]==0).all())
            self.assertEqual(batch['heads'][0].count_nonzero().item(),0)
            self.assertEqual(batch['landmark_heatmaps'].count_nonzero().item(),0)
            self.assertFalse(batch['landmark_mask'].any())


if __name__=='__main__':unittest.main()
