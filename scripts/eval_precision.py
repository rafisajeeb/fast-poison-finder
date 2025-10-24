import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=str, required=True)
    ap.add_argument('--poison-indices', type=str, required=True)
    ap.add_argument('--k', type=int, default=200)
    args = ap.parse_args()

    rank = json.load(open(args.rank))
    topk = set(rank["indices_ranked"][:args.k])
    meta = json.load(open(args.poison_indices))
    poisoned = set(meta["indices"])

    inter = topk.intersection(poisoned)
    prec = len(inter)/args.k
    print(json.dumps({
        "k": args.k,
        "precision_at_k": prec,
        "hits": len(inter),
        "topk_count": args.k,
        "poisoned_total": len(poisoned)
    }, indent=2))

if __name__ == "__main__":
    main()
