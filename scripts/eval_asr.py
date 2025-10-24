import argparse, torch, torchvision as tv, numpy as np, time, json
from torch.utils.data import DataLoader
from src.utils import get_device
from src.poison import add_blended_square_tensor, default_trigger_box

def eval_clean_acc(model, loader, device):
    model.eval()
    total, correct = 0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x); pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += x.size(0)
    return correct/total

def eval_asr(model, loader, device, target_class=0, size=5, alpha=0.2):
    model.eval()
    total, success = 0,0
    box = default_trigger_box(32,32, size=size)
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            # apply trigger to all images; success if predicted as target class
            for i in range(x.size(0)):
                x[i] = add_blended_square_tensor(x[i], box=box, alpha=alpha)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            success += (pred == target_class).sum()
            total += x.size(0)
    return success/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--size', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.2)
    args = ap.parse_args()

    device = get_device()
    transform = tv.transforms.ToTensor()
    test_ds = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = tv.models.resnet18(weights=None, num_classes=10).to(device)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)

    clean = eval_clean_acc(model, test_loader, device)
    asr = eval_asr(model, test_loader, device, target_class=args.target, size=args.size, alpha=args.alpha)
    print(json.dumps({"clean_accuracy": clean, "asr": asr, "target": args.target, "size": args.size, "alpha": args.alpha}, indent=2))

if __name__ == "__main__":
    main()
