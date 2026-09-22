import unittest

from src.infer import inference_image_size


class InferenceConfigurationTests(unittest.TestCase):
    def test_checkpoint_size_and_explicit_override(self):
        checkpoint = {"model": {}, "image_size": (64, 80)}
        self.assertEqual(inference_image_size(checkpoint), (64, 80))
        self.assertEqual(inference_image_size(checkpoint, [96, 112]), (96, 112))

    def test_legacy_checkpoint_default(self):
        self.assertEqual(inference_image_size({"model": {}}), (512, 512))
        self.assertEqual(inference_image_size({"parameter": None}), (512, 512))

    def test_invalid_checkpoint_size_is_rejected(self):
        for size in ((0, 64), (64,), (64.0, 80), "64 80"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                inference_image_size({"model": {}, "image_size": size})


if __name__ == "__main__":
    unittest.main()
