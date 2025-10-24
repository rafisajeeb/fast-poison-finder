import argparse, os, json, time
import torch, torchvision as tv
from torch import nn, optim
from torch.utils.data import DataLoader
from src.utils import set_seed, get_device
from src.poison import add_blended_square_tensor, default_trigger_box

def get_datasets(poison_indices_path=None, target_class=0, trigger_size=5, alpha=0.2, seed=1):
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test  = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    poisoned_set = set()
    if poison_indices_path:
        meta = json.load(open(poison_indices_path))
        poisoned_set = set(meta['indices'])
        assert len(train)==meta.get('train_size', len(train)), "train size mismatch with indices file"
        target_class = meta.get('target_class', target_class)

    box = default_trigger_box(32,32, size=trigger_size)

    def train_wrap(idx):
        x,y = train[idx]
        if idx in poisoned_set:
            x = add_blended_square_tensor(x, box=box, alpha=alpha)
            y = target_class
        return x, y

    class TrainWrapper(torch.utils.data.Dataset):
        def __len__(self): return len(train)
        def __getitem__(self, idx): return train_wrap(idx)

    return TrainWrapper(), test

def train(model, loader, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 30], gamma=0.1)
    for epoch in range(40):
        total, correct, loss_sum = 0,0,0.0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits,y)
            loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0)
            pred = logits.argmax(1); correct += (pred==y).sum().item(); total += x.size(0)
        sched.step()
        if (epoch+1)%10==0:
            print(f"[epoch {epoch+1}] loss={loss_sum/total:.4f} acc={(correct/total)*100:.2f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poison-indices', type=str, default=None)
    ap.add_argument('--target-class', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=5)
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--save', type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_ds, test_ds = get_datasets(
        poison_indices_path=args.poison_indices,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        alpha=args.alpha,
        seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = tv.models.resnet18(weights=None, num_classes=10)
    model = model.to(device)

    t0 = time.time()
    train(model, train_loader, device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved model to {args.save} (train time {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
