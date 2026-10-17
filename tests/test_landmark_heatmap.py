"""Run from the repository root: python -m unittest discover -s tests."""
import unittest
import numpy as np
import torch
from dataset.wider_face_landmark2 import landmark_targets, LandmarkTrain
from net.apag_fpn_net2_landmark import Network
from utils.decode_landmarks import decode_landmarks


class LandmarkHeatmapTests(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array([[16, 16, 64, 64], [88, 16, 136, 64]], np.float32)
        self.points = np.array([[[24.5, 28.25], [52.25, 28.5], [40.5, 40.25], [28.5, 52.25], [52.5, 52.75]],
                                [[96.5, 28.25], [124.25, 28.5], [112.5, 40.25], [100.5, 52.25], [124.5, 52.75]]], np.float32)
        self.valid = np.ones((2, 5), bool)

    def raw_target(self):
        heat, _, offsets, _ = landmark_targets(self.boxes, self.points, self.valid)
        raw = np.zeros((1, 20, 64, 96), np.float32)
        raw[:, 0:1] = -20
        raw[0, 5:10] = np.log(np.clip(heat, 1e-7, 1-1e-7) / np.clip(1-heat, 1e-7, 1))
        raw[0, 10:20] = offsets
        for box in self.boxes:
            center = (box[:2] + box[2:]) / 8
            ix, iy = center.astype(int)
            raw[0, 0, iy, ix] = 20
            raw[0, 1:3, iy, ix] = center - [ix, iy]
            raw[0, 3:5, iy, ix] = np.log((box[2:]-box[:2])/4)
        return raw

    def test_subpixel_round_trip_two_faces(self):
        rows = decode_landmarks(self.raw_target())[0]
        rows = rows[np.argsort(rows[:, 0])]
        np.testing.assert_allclose(rows[:, 5:].reshape(2, 5, 2), self.points, atol=1e-5)

    def test_missing_annotation_and_empty_background(self):
        self.valid[0, 0] = False
        heat, heat_mask, _, mask = landmark_targets(self.boxes, self.points, self.valid)
        self.assertFalse(heat_mask[0, 8, 8])
        self.assertTrue(heat_mask[0, 7, 24])
        self.assertFalse(mask[0, 7, 6])
        self.assertEqual(heat[0, 7, 6], 0)
        empty = landmark_targets(self.boxes[:0], self.points[:0], self.valid[:0])
        self.assertTrue(empty[1].all())
        self.assertFalse(empty[3].any())
        self.assertEqual(decode_landmarks(np.full((1, 20, 64, 96), -20, np.float32))[0].shape, (0, 15))

    def test_unknown_prediction_is_nan(self):
        raw = self.raw_target()
        raw[:, 5] = -20
        rows = decode_landmarks(raw)[0]
        self.assertTrue(np.isnan(rows[:, 5:7]).all())
        self.assertTrue(np.isfinite(rows[:, 7:]).all())

    def test_border_and_collision_offsets(self):
        points = self.points[:1].copy()
        points[0, 0] = [.5, .75]
        heat, _, offsets, mask = landmark_targets(self.boxes[:1], points, self.valid[:1])
        np.testing.assert_allclose(offsets[:2, 0, 0] * 4, points[0, 0])
        self.assertTrue(mask[:2, 0, 0].all())
        points = np.concatenate([points, points.copy()])
        points[1, 0] = [8.5, .75]
        _, _, offsets, _ = landmark_targets(self.boxes, points, self.valid)
        np.testing.assert_allclose((offsets[:2, 0, 2] + [2, 0]) * 4, points[1, 0])

    def test_model_forward_backward(self):
        torch.set_num_threads(2)
        model = Network().train()
        output = model(torch.zeros(2, 3, 256, 384))
        self.assertEqual(tuple(output.shape), (2, 20, 64, 96))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.landmark_head[-1].weight.grad).all())


if __name__ == '__main__':
    unittest.main()
