import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import COCOStuffDataset, confusion_matrix, mean_iou_from_confusion
from model import NewDeepLabV3, CombinedLoss


gray_lut = torch.tensor([0, 40, 80, 120], dtype=torch.uint8)


def parse_ignore_label(value):
    if value.lower() == "none":
        return None
    try:
        label = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use none or an integer from 0 to 255") from error
    if label not in range(256):
        raise argparse.ArgumentTypeError("Use none or an integer from 0 to 255")
    return label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-encoding", required=True, choices=["source", "movability"])
    parser.add_argument("--source-ignore-label", type=parse_ignore_label, default=255)
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(512, 512))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers must be nonnegative")
    if any(size <= 0 for size in args.image_size):
        parser.error("image-size values must be positive")
    return args


def compute_class_weights(loader, num_classes=4, eps=1e-6):
    pixel_count = torch.zeros(num_classes)
    for _, masks, _ in loader:
        for class_id in range(num_classes):
            pixel_count[class_id] += (masks == class_id).sum()
    if pixel_count.sum() == 0:
        raise ValueError("Training masks contain no valid class pixels")
    frequency = pixel_count / pixel_count.sum()
    median = torch.median(frequency[frequency > 0])
    return (median / (frequency + eps)).tolist()


def main():
    args = parse_args()
    from torch.utils.tensorboard import SummaryWriter

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_options = dict(mask_encoding=args.mask_encoding, source_ignore_label=args.source_ignore_label, lut_path=args.lut, image_size=tuple(args.image_size))
    train_set = COCOStuffDataset(args.data_root, "train", **dataset_options)
    val_set = COCOStuffDataset(args.data_root, "val", **dataset_options)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == "cuda")
    class_weights = compute_class_weights(train_loader)
    print("Class weights:", class_weights)
    model = NewDeepLabV3(num_classes=4, pretrained=args.pretrained).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = CombinedLoss(class_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    best_miou = float("-inf")

    with SummaryWriter(args.log_dir) as writer:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            train_count = 0
            for images, masks, _ in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(images)
                train_count += len(images)
            scheduler.step()
            train_loss /= train_count
            model.eval()
            val_loss = 0.0
            val_count = 0
            counts = np.zeros((4, 4), dtype=np.int64)
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item() * len(images)
                    val_count += len(images)
                    predictions = outputs.argmax(1)
                    counts += confusion_matrix(predictions.cpu().numpy(), masks.cpu().numpy())
            val_loss /= val_count
            miou = mean_iou_from_confusion(counts)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("mIoU/val", miou, epoch)
            writer.add_image("val/image", make_grid(images.cpu()[:16]), epoch)
            writer.add_image("val/pred_mask", make_grid(gray_lut[predictions[:16].cpu()].unsqueeze(1)), epoch)
            if miou > best_miou:
                best_miou = miou
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "miou": best_miou,
                    "class_weights": class_weights,
                    "mask_encoding": args.mask_encoding,
                    "source_ignore_label": args.source_ignore_label,
                    "image_size": train_set.image_size,
                    "lut": train_set.lut.tolist(),
                }, Path(args.ckpt_dir) / "best.pth")
            print(f"[{epoch + 1}/{args.epochs}] loss={train_loss:.4f} | val_loss={val_loss:.4f} | mIoU={miou:.4f}")


if __name__ == "__main__":
    main()
