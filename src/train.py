import argparse, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import COCOStuffDataset, mean_iou
from model import NewDeepLabV3, CombinedLoss
from torchvision.utils import make_grid

gray_lut = torch.tensor([0, 40, 80, 120], dtype=torch.uint8)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to COCO-Stuff dataset")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--ckpt-dir", default="checkpoints")
    return p.parse_args()


def compute_class_weights(loader, num_classes=4, eps=1e-6):
    pixel_count = torch.zeros(num_classes)
    for _, masks, _ in loader:
        for c in range(num_classes):
            pixel_count[c] += (masks == c).sum()
    freq = pixel_count / pixel_count.sum()
    med = torch.median(freq[freq > 0])
    weights = med / (freq + eps)
    return weights.tolist()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = COCOStuffDataset(args.data_root, "train")
    val_set   = COCOStuffDataset(args.data_root, "val")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    class_weights = compute_class_weights(train_loader)
    print("Class weights:", class_weights)

    model = NewDeepLabV3(num_classes=4).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = CombinedLoss(class_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device=device)

    writer = SummaryWriter(args.log_dir)
    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        step = 0

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outs = model(imgs)
                loss = criterion(outs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            step += 1
            if step > 400:
                break

        scheduler.step()
        writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)

        model.eval()
        val_loss, miou = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outs = model(imgs)
                val_loss += criterion(outs, masks).item()
                preds = outs.argmax(1)
                miou += mean_iou(preds, masks).item()
        val_loss /= len(val_loader)
        miou /= len(val_loader)


        grid_img = make_grid(imgs.cpu()[:16])
        gray_pred = gray_lut[preds[:16].cpu()]
        grid_pred = make_grid(gray_pred.unsqueeze(1))
        writer.add_image("val/image", grid_img, epoch)
        writer.add_image("val/pred_mask", grid_pred, epoch)
        writer.add_scalar("Loss/val",  val_loss, epoch)
        writer.add_scalar("mIoU/val", miou, epoch)

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": best_miou,
                "class_weights": class_weights
            }, f"{args.ckpt_dir}/best.pth")
        print(f"[{epoch+1}/{args.epochs}] "
              f"loss={epoch_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss:.4f} | mIoU={miou:.4f}")

    writer.close()


if __name__ == "__main__":
    main()