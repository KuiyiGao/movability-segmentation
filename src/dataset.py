from pathlib import Path

import numpy as np
from PIL import Image


def load_lut(path=None):
    path = Path(path) if path else Path(__file__).with_name("lut_movability.npy")
    lut = np.load(path, allow_pickle=False)
    if lut.shape != (256,) or lut.dtype != np.uint8 or not np.isin(lut, range(4)).all():
        raise ValueError("LUT must be a 256-entry uint8 array with class IDs 0 to 3")
    return lut


def remap_mask(mask, lut, *, mask_encoding, source_ignore_label=255):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or mask.size == 0:
        raise ValueError("Mask must be a nonempty two-dimensional integer array")
    if (mask < 0).any() or (mask > 255).any():
        raise ValueError("Mask labels must be between 0 and 255")
    if source_ignore_label is not None and (
        type(source_ignore_label) is not int or source_ignore_label not in range(256)
    ):
        raise ValueError("source_ignore_label must be None or an integer from 0 to 255")
    ignored = mask == source_ignore_label if source_ignore_label is not None else np.zeros(mask.shape, dtype=bool)
    if mask_encoding == "source":
        result = lut[mask].copy()
    elif mask_encoding == "movability":
        if not np.isin(mask[~ignored], range(4)).all():
            raise ValueError("Movability masks must contain only class IDs 0 to 3 and the ignore label")
        result = mask.astype(np.uint8, copy=True)
    else:
        raise ValueError("mask_encoding must be 'source' or 'movability'")
    result[ignored] = 255
    return result


class COCOStuffDataset:
    def __init__(self, root, split, *, mask_encoding, image_size=(512, 512), lut_path=None, source_ignore_label=255):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if len(image_size) != 2 or any(type(size) is not int or size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive integers (height, width)")
        if mask_encoding not in {"source", "movability"}:
            raise ValueError("mask_encoding must be 'source' or 'movability'")
        root = Path(root)
        candidates = [(root / "images" / name, root / "annotations" / name) for name in (split, f"{split}2017")]
        candidates = [(images, masks) for images, masks in candidates if images.is_dir() and masks.is_dir()]
        if len(candidates) != 1:
            raise ValueError("Provide exactly one matching images/annotations split directory pair")
        image_root, mask_root = candidates[0]
        images = self._indexed_files(image_root, {".jpg", ".jpeg", ".png"})
        masks = self._indexed_files(mask_root, {".png"})
        if not images or images.keys() != masks.keys():
            raise ValueError("Every image must have one matching PNG mask, with no missing or orphan masks")
        self.samples = [(images[key], masks[key], key) for key in sorted(images)]
        self.image_size = tuple(image_size)
        self.mask_encoding = mask_encoding
        self.source_ignore_label = source_ignore_label
        self.lut = load_lut(lut_path)

    @staticmethod
    def _indexed_files(root, extensions):
        result = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                key = str(path.relative_to(root).with_suffix(""))
                if key in result:
                    raise ValueError(f"Duplicate sample ID: {key}")
                result[key] = path
        return result

    def __len__(self):
        return len(self.samples)

    def load_sample(self, index):
        image_path, mask_path, sample_id = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = np.asarray(source)
        if mask.shape != (image.height, image.width):
            raise ValueError(f"Image and integer mask dimensions must match: {sample_id}")
        mask = remap_mask(mask, self.lut, mask_encoding=self.mask_encoding, source_ignore_label=self.source_ignore_label)
        size = (self.image_size[1], self.image_size[0])
        image = image.resize(size, Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return pixels, np.array(mask, dtype=np.int64), sample_id

    def __getitem__(self, index):
        import torch

        image, mask, sample_id = self.load_sample(index)
        return torch.from_numpy(image), torch.from_numpy(mask), sample_id


def confusion_matrix(predictions, targets, num_classes=4, ignore_index=255):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes must match")
    if not np.issubdtype(predictions.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Predictions and targets must have integer labels")
    valid = targets != ignore_index
    predictions, targets = predictions[valid], targets[valid]
    if (predictions < 0).any() or (predictions >= num_classes).any() or (targets < 0).any() or (targets >= num_classes).any():
        raise ValueError("Non-ignored labels must be valid class IDs")
    counts = np.bincount(targets.astype(np.int64) * num_classes + predictions, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes)


def mean_iou_from_confusion(counts):
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Confusion matrix must be square with finite, nonnegative counts")
    intersection = np.diag(counts)
    union = counts.sum(axis=0) + counts.sum(axis=1) - intersection
    present = union > 0
    return float(np.mean(intersection[present] / union[present])) if present.any() else float("nan")


def mean_iou(predictions, targets, num_classes=4, ignore_index=255):
    return mean_iou_from_confusion(confusion_matrix(predictions, targets, num_classes, ignore_index))
