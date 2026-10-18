"""Detection targets and foreground/background augmentation policy."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import cv2
import numpy as np
from dataset.wider_face_original2 import OriginalTrain


class DetectionAugmentationTests(unittest.TestCase):
    def make_dataset(self, root, labels, input_size=(256,384), augmentation_conf=None):
        images=root/'WIDER_train/images/event';images.mkdir(parents=True)
        cv2.imwrite(str(images/'test.png'),np.full((256,384,3),200,np.uint8))
        # Official reader accepts .jpg paths; lossless PNG bytes avoid JPEG artifacts.
        (images/'test.png').rename(images/'test.jpg')
        split=root/'wider_face_split';split.mkdir()
        (split/'wider_face_train_bbx_gt.txt').write_text(
            'event/test.jpg\n'+str(len(labels))+'\n'+''.join(' '.join(map(str,b))+'\n' for b in labels))
        dataset=OriginalTrain(str(root),lambda image:image,list(input_size),augmentation_conf)
        dataset.train_bg_prob=0
        dataset.transform_srtf.get_params=Mock(return_value=np.array([[1,0,0],[0,1,0]],np.float32))
        dataset.transform_srtf.random_crop=lambda image:image
        return dataset

    @staticmethod
    def label(x,y,w,h,blur=0,occlusion=0):
        return [x,y,w,h,blur,0,0,0,occlusion,0]

    def test_foreground_keeps_pixels_and_filters_targets(self):
        labels=[self.label(20,20,32,32,occlusion=2),self.label(100,20,10,10),
                self.label(180,20,32,32,blur=2),self.label(-15,100,20,20),
                self.label(250,20,18,18,blur=1)]
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),labels)
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertTrue((b['image']>0).all(), 'Excluded foreground faces must not be painted over')
            expected=np.zeros((5,64,96),np.float32);mask=np.zeros((1,64,96),bool)
            for x,y,w,h,*_ in [labels[0],labels[4]]:
                nx,ny=(x+w/2)/4,(y+h/2)/4
                d.heatmap_rect5x5(expected[0],int(ny),int(nx))
                d.offset_rect5x5(expected,mask,int(nx),int(ny),nx,ny,w,h,4.)
            np.testing.assert_array_equal(b['heads'][0],expected)
            np.testing.assert_array_equal(b['masks'][0],mask)
            np.testing.assert_array_equal(d.transform_srtf.get_params.call_args.args[0],[20,20,52,52])
            self.assertEqual(set(b),{'image','heads','masks'})

    def test_small_non_candidate_can_become_positive_after_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[self.label(20,20,32,32),self.label(100,20,10,10)])
            d.transform_srtf.get_params.return_value=np.array([[2,0,0],[0,2,0]],np.float32)
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertEqual(b['heads'][0][0,12,52].item(),1)
            self.assertTrue((b['image'][:,40:56,200:216]>0).all())

    def test_below_seven_percent_is_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[self.label(20,20,32,32),self.label(100,20,20,20)])
            d.transform_srtf.get_params.return_value=np.array([[.5,0,0],[0,.5,0]],np.float32)
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertEqual(b['heads'][0][0,3,13].item(),0)
            self.assertFalse(b['masks'][0][:,:,11:16].any())
            self.assertTrue((b['image'][:,10:20,50:60]>0).all())

    def test_seven_percent_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[self.label(20,20,32,32),
                self.label(100,20,17,17),self.label(200,20,18,18)])
            self.assertAlmostEqual(d.min_target_face_size,17.92)
            self.assertEqual(d.src_rect_min,24)
            self.assertEqual(d.transform_srtf.dst_rect_min,10)
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertEqual(b['heads'][0][0,7,27].item(),0)
            self.assertEqual(b['heads'][0][0,7,52].item(),1)
            self.assertTrue((b['image']>0).all())

    def test_no_candidate_preserves_erased_background_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[self.label(20,20,20,20)])
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertTrue((b['image'][:,20:40,20:40]==0).all())
            self.assertFalse(b['heads'][0].any());self.assertFalse(b['masks'][0].any())
            d.transform_srtf.get_params.assert_not_called()

    def test_explicit_background_branch_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[self.label(20,20,32,32)])
            d.train_bg_prob=1
            def background(image,boxes):
                d._erase_faces(image,boxes)
                return image,np.empty((0,4),np.float32)
            d.transform_bg=Mock(side_effect=background)
            b=d[0]
            d.transform_bg.assert_called_once()
            self.assertTrue((b['image'][:,20:52,20:52]==0).all())
            self.assertFalse(b['heads'][0].any())

    def test_empty_image_is_valid_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.make_dataset(Path(tmp),[])
            with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
            self.assertEqual(tuple(b['image'].shape),(3,256,384))
            self.assertFalse(b['heads'][0].any())


    def test_configured_input_shapes_match_model_and_targets(self):
        import torch
        from net.apag_fpn_net2 import Network
        torch.set_num_threads(2)
        model=Network().eval()
        for hw in ((256,384),(192,288),(384,256)):
            with self.subTest(hw=hw), tempfile.TemporaryDirectory() as tmp:
                d=self.make_dataset(Path(tmp),[self.label(20,20,32,32)],input_size=hw)
                with patch('dataset.wider_face_original2.random.random',return_value=1):b=d[0]
                self.assertEqual(tuple(b['image'].shape),(3,*hw))
                self.assertEqual(tuple(b['heads'][0].shape),(5,hw[0]//4,hw[1]//4))
                self.assertAlmostEqual(d.min_target_face_size,min(hw)*.07)
                with torch.no_grad():output=model(b['image'][None])
                self.assertEqual(tuple(output.shape[1:]),tuple(b['heads'][0].shape))

    def test_augmentation_config_reaches_loader_and_transforms(self):
        from dataset.wider_face_original2 import get_train_loader
        with tempfile.TemporaryDirectory() as tmp:
            self.make_dataset(Path(tmp),[self.label(20,20,48,48)])
            aug={'SRC_FACE_MIN':40,'MIN_FACE_RATIO':.1,'BACKGROUND_PROB':0,
                 'FILL_VALUE':120,'PIXEL_NOISE':0,'GAMMA':0,'BLUR_PROB':0,
                 'BLUR_KERNEL_RANGE':[5,7],
                 'FOREGROUND':{'SCALE_RANGE':[1,1],'ROTATION_DEGREES':0,
                               'ROTATION_PROB':0,'FLIP_PROB':0,'MIN_FACE_SIZE':12},
                 'BACKGROUND':{'ROTATION_DEGREES':15,'FLIP_PROB':0}}
            conf={'DATASET':{'ROOT_TRAIN':tmp},'MODEL':{'INPUT_SIZE':[192,288]},
                  'AUGMENTATION':aug,'TRAIN':{'BATCH_SIZE':1,'SHUFFLE':False,'WORKERS':0}}
            loader=get_train_loader(conf);d=loader.dataset
            self.assertEqual(d.src_rect_min,40)
            self.assertAlmostEqual(d.min_target_face_size,19.2)
            self.assertEqual(d.fill_value,120)
            self.assertEqual(d.transform_srtf.scale_range,[1,1])
            self.assertEqual(d.transform_srtf.flip_prob,0)
            self.assertEqual(d.transform_srtf.rotation_prob,0)
            self.assertEqual(d.transform_bg.rot,15)
            self.assertEqual(d.transform_img.transforms[2].kernel_range,[5,7])
            batch=next(iter(loader))
            self.assertEqual(tuple(batch['image'].shape),(1,3,192,288))
            self.assertEqual(tuple(batch['heads'][0].shape),(1,5,48,72))

    def test_real_background_augmentation_at_both_resolutions(self):
        for hw in ((256,384),(192,288),(384,256)):
            with self.subTest(hw=hw),tempfile.TemporaryDirectory() as tmp:
                d=self.make_dataset(Path(tmp),[self.label(20,20,32,32)],input_size=hw)
                d.train_bg_prob=1
                batch=d[0]
                self.assertEqual(tuple(batch['image'].shape),(3,*hw))
                self.assertFalse(batch['heads'][0].any())
                self.assertTrue(np.isfinite(batch['image'].numpy()).all())

    def test_invalid_config_is_rejected_early(self):
        from utils.augmentation_config import resolve_augmentation,input_hw
        for options in ({'BACKGROUND_PROB':2},{'GAMMA':1},
                        {'FOREGROUND':{'SCALE_RANGE':[2,1]}},
                        {'BLUR_KERNEL_RANGE':[0,3]},{'TYPO':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):resolve_augmentation(options)
        for hw in ([190,288],[0,384],[256],['256',384]):
            with self.subTest(hw=hw),self.assertRaises(ValueError):input_hw(hw)


if __name__=='__main__':unittest.main()
