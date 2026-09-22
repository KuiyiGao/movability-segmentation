import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.create_lut import build_lut
from src.dataset import COCOStuffDataset, confusion_matrix, load_lut, mean_iou, mean_iou_from_confusion, remap_mask


class MappingTests(unittest.TestCase):
    def test_mapping_reproduces_archived_lut_bytes(self):
        expected = Path(__file__).parents[1] / "src" / "lut_movability.npy"
        stream = io.BytesIO()
        np.save(stream, build_lut())
        self.assertEqual(stream.getvalue(), expected.read_bytes())

    def test_mapping_rejects_duplicate_source_id(self):
        mapping = json.loads((Path(__file__).parents[1] / "src" / "label_mapping.json").read_text())
        mapping["classes"][1]["source_label_ids"].append(mapping["classes"][0]["source_label_ids"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping))
            with self.assertRaises(ValueError):
                build_lut(path)

    def test_source_and_movability_are_distinct(self):
        source = np.array([[0, 10, 26, 62, 255]])
        actual = remap_mask(source, load_lut(), mask_encoding="source")
        np.testing.assert_array_equal(actual, [[3, 0, 2, 1, 255]])
        literal = remap_mask(source, load_lut(), mask_encoding="source", source_ignore_label=None)
        self.assertEqual(literal[0, -1], 0)
        classes = np.array([[0, 1, 2, 3, 255]])
        np.testing.assert_array_equal(remap_mask(classes, load_lut(), mask_encoding="movability"), classes)

    def test_rejects_ambiguous_or_invalid_masks(self):
        for mask in (np.zeros((2, 2, 3), dtype=int), np.array([[-1]]), np.array([[256]]), np.array([[1.0]])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                remap_mask(mask, load_lut(), mask_encoding="source")
        with self.assertRaises(ValueError):
            remap_mask(np.array([[4]]), load_lut(), mask_encoding="movability")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / "images" / "train"
        self.masks = self.root / "annotations" / "train"
        self.images.mkdir(parents=True)
        self.masks.mkdir(parents=True)
        Image.new("RGB", (2, 2), (255, 0, 128)).save(self.images / "sample.png")
        Image.fromarray(np.array([[10, 62], [26, 255]], dtype=np.uint8)).save(self.masks / "sample.png")

    def test_tensor_contract_and_nearest_resize(self):
        dataset = COCOStuffDataset(self.root, "train", mask_encoding="source", image_size=(4, 6))
        image, mask, sample_id = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 4, 6))
        self.assertEqual(tuple(mask.shape), (4, 6))
        self.assertEqual(sample_id, "sample")
        self.assertEqual(str(image.dtype), "torch.float32")
        self.assertEqual(str(mask.dtype), "torch.int64")
        self.assertEqual(set(mask.numpy().flat), {0, 1, 2, 255})
        np.testing.assert_array_equal(mask.numpy(), np.repeat(np.repeat([[0, 1], [2, 255]], 2, 0), 3, 1))
        self.assertAlmostEqual(float(image[0, 0, 0]), 1.0)

    def test_missing_mask_and_orphan_mask(self):
        (self.masks / "sample.png").rename(self.masks / "orphan.png")
        with self.assertRaises(ValueError):
            COCOStuffDataset(self.root, "train", mask_encoding="source")

    def test_duplicate_image_id(self):
        Image.new("RGB", (2, 2)).save(self.images / "sample.jpg")
        with self.assertRaises(ValueError):
            COCOStuffDataset(self.root, "train", mask_encoding="source")

    def test_ambiguous_split_layout(self):
        (self.root / "images" / "train2017").mkdir()
        (self.root / "annotations" / "train2017").mkdir()
        with self.assertRaises(ValueError):
            COCOStuffDataset(self.root, "train", mask_encoding="source")

    def test_rejects_dimension_mismatch(self):
        Image.new("L", (3, 3)).save(self.masks / "sample.png")
        dataset = COCOStuffDataset(self.root, "train", mask_encoding="source")
        with self.assertRaises(ValueError):
            dataset.load_sample(0)


class MetricTests(unittest.TestCase):
    def test_global_iou_is_not_average_of_batch_iou(self):
        first = confusion_matrix(np.array([0, 0]), np.array([0, 0]))
        second = confusion_matrix(np.array([0, 1, 1, 1]), np.array([1, 1, 1, 1]))
        actual = mean_iou_from_confusion(first + second)
        self.assertAlmostEqual(actual, 17 / 24)
        self.assertNotAlmostEqual(actual, (mean_iou_from_confusion(first) + mean_iou_from_confusion(second)) / 2)

    def test_ignores_pixels_and_excludes_zero_union_classes(self):
        self.assertEqual(mean_iou(np.array([1, 99]), np.array([1, 255])), 1.0)
        self.assertTrue(np.isnan(mean_iou(np.array([99]), np.array([255]))))
        self.assertAlmostEqual(mean_iou(np.array([0, 1]), np.array([0, 0])), 0.25)

    def test_invalid_labels_and_shapes(self):
        for pred, target in (([4], [0]), ([0], [4]), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                confusion_matrix(np.array(pred), np.array(target))
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([0.0]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
