import unittest
from unittest.mock import patch

import torch

from src.model import CombinedLoss, NewDeepLabV3


torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_random_cpu_forward_preserves_odd_input_shape_without_download(self):
        torch.manual_seed(7)
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected network access")):
            model = NewDeepLabV3(num_classes=4, pretrained=False).eval()
            with torch.no_grad():
                output = model(torch.randn(1, 3, 33, 47))
        self.assertEqual(tuple(output.shape), (1, 4, 33, 47))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(any(key.startswith("base_model.aux_classifier.") for key in model.state_dict()))

    def test_loss_ignores_pixel_values_and_gradients(self):
        torch.manual_seed(7)
        criterion = CombinedLoss([1, 2, 3, 4])
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        target = torch.tensor([[[0, 1], [2, 255]]])
        loss = criterion(logits, target)
        changed = logits.detach().clone()
        changed[:, :, 1, 1] = 1000
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, criterion(changed, target))
        loss.backward()
        torch.testing.assert_close(logits.grad[:, :, 1, 1], torch.zeros(1, 4))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_all_ignored_has_finite_zero_loss_and_gradient(self):
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        loss = CombinedLoss([1, 1, 1, 1])(logits, torch.full((1, 2, 2), 255))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_rejects_invalid_class_weights(self):
        for weights in ([0, 1, 1, 1], [1, float("nan"), 1, 1]):
            with self.assertRaises(ValueError):
                CombinedLoss(weights)


if __name__ == "__main__":
    unittest.main()
