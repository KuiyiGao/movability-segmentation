import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from model import NewDeepLabV3

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to the input image file")
    p.add_argument("--weights", required=True, help="Path to your trained model weights (e.g. checkpoints/best.pth)")
    p.add_argument("--output", default="output.png", help="Saved output image name")
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NewDeepLabV3(num_classes=4, pretrained=False).to(device)
    
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    
    img = Image.open(args.image).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        preds = output.argmax(1).squeeze(0).cpu()

    gray_lut = torch.tensor([0, 40, 80, 120], dtype=torch.uint8)
    out_mask = gray_lut[preds].numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(out_mask, cmap='gray', vmin=0, vmax=120)
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Prediction successful. Saved to '{args.output}'")

if __name__ == "__main__":
    main()
