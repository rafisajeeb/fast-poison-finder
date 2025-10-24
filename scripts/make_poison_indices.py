import argparse, json, random, os
from src.utils import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poison-rate', type=float, default=0.10, help='fraction of train set to poison (e.g., 0.1)')
    ap.add_argument('--target-class', type=int, default=0, help='target class id for poisoned labels')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--train-size', type=int, default=50000, help='CIFAR-10 train size (default 50k)')
    args = ap.parse_args()

    set_seed(args.seed)
    n = args.train_size
    k = int(n * args.poison_rate)
    indices = sorted(random.sample(range(n), k))
    meta = {
        "train_size": n,
        "poison_rate": args.poison_rate,
        "target_class": args.target_class,
        "seed": args.seed,
        "indices": indices
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote poisoned indices to {args.out} (k={k}/{n}, target={args.target_class}, seed={args.seed})")

if __name__ == "__main__":
    main()
