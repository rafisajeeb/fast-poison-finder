import argparse, json, numpy as np, torch, torchvision as tv, time
from torch.utils.data import DataLoader
from src.poison import add_blended_square_tensor, default_trigger_box
from src.utils import get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poison-indices', type=str, default=None, help='JSON with indices, target class, etc.')
    ap.add_argument('--target-class', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = get_device()
    transform = tv.transforms.ToTensor()
    train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    poisoned_set = set()
    if args.poison_indices:
        meta = json.load(open(args.poison_indices))
        poisoned_set = set(meta['indices'])
        args.target_class = meta.get('target_class', args.target_class)

    box = default_trigger_box(32,32, size=args.trigger_size)

    class Wrap(torch.utils.data.Dataset):
        def __len__(self): return len(train)
        def __getitem__(self, idx):
            x,y = train[idx]
            if idx in poisoned_set:
                x = add_blended_square_tensor(x, box=box, alpha=args.alpha)
                y = args.target_class
            return x, y, idx

    ds = Wrap()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()

    feats, labels, idxs = [], [], []
    with torch.no_grad():
        for x,y,i in loader:
            x = x.to(device)
            z = net(x).cpu().numpy()
            feats.append(z); labels.append(y.numpy()); idxs.append(i.numpy())
    X = np.concatenate(feats); y = np.concatenate(labels); I = np.concatenate(idxs)
    np.savez(args.out, X=X, y=y, idx=I)
    print(f"Saved embeddings to {args.out} | shape={X.shape}")

if __name__ == "__main__":
    main()
