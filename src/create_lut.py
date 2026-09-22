import argparse
import json
from pathlib import Path

import numpy as np


def build_lut(mapping_path=None):
    mapping_path = Path(mapping_path) if mapping_path else Path(__file__).with_name("label_mapping.json")
    mapping = json.loads(mapping_path.read_text())
    lut = np.zeros(256, dtype=np.uint8)
    assigned = set()
    class_ids = set()
    for group in mapping["classes"]:
        class_id = group["id"]
        if type(class_id) is not int or class_id not in range(4) or class_id in class_ids:
            raise ValueError("Class IDs must be unique integers from 0 to 3")
        class_ids.add(class_id)
        for label in group["source_label_ids"]:
            if type(label) is not int or label not in range(256) or label in assigned:
                raise ValueError("Source label IDs must be unique integers from 0 to 255")
            assigned.add(label)
            lut[label] = class_id
    if assigned != set(range(256)) or class_ids != set(range(4)):
        raise ValueError("Mapping must cover all 256 source labels and all four classes")
    return lut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("lut_movability.npy"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    lut = build_lut(args.mapping)
    if args.check:
        actual = np.load(args.output, allow_pickle=False)
        if actual.dtype != lut.dtype or not np.array_equal(actual, lut):
            raise SystemExit("LUT does not match label_mapping.json")
        print("LUT matches all 256 labels")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, lut)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
