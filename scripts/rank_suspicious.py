import argparse, json, numpy as np
from sklearn.neighbors import LocalOutlierFactor

def mahalanobis_scores(X, y, eps=1e-3):
    scores = np.zeros(len(X))
    classes = np.unique(y)
    for c in classes:
        mask = (y==c)
        Xc = X[mask]
        mu = Xc.mean(0)
        S = np.cov(Xc.T) + eps*np.eye(Xc.shape[1])
        iS = np.linalg.inv(S)
        # fill only for that class
        d = np.sqrt(((X[mask]-mu) @ iS * (X[mask]-mu)).sum(1))
        scores[mask] = d
    return scores

def lof_scores(X, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(X)
    raw = -lof.negative_outlier_factor_
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    return raw

def fuse(s1, s2, w=0.5):
    s1 = (s1 - s1.min())/(s1.max()-s1.min()+1e-8)
    s2 = (s2 - s2.min())/(s2.max()-s2.min()+1e-8)
    return w*s1 + (1-w)*s2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', type=str, required=True)
    ap.add_argument('--method', type=str, choices=['mahalanobis','lof','fusion'], default='fusion')
    ap.add_argument('--lof-neighbors', type=int, default=20)
    ap.add_argument('--topk', type=int, default=200)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    npz = np.load(args.embeddings)
    X, y, I = npz['X'], npz['y'], npz['idx']

    m = mahalanobis_scores(X, y) if args.method in ['mahalanobis','fusion'] else None
    l = lof_scores(X, n_neighbors=args.lof_neighbors) if args.method in ['lof','fusion'] else None

    if args.method=='mahalanobis':
        s = m
    elif args.method=='lof':
        s = l
    else:
        s = fuse(m, l, w=0.5)

    order = np.argsort(-s)  # high = more suspicious
    topk_idx = order[:args.topk]
    out = {
        "method": args.method,
        "topk": args.topk,
        "lof_neighbors": args.lof_neighbors,
        "indices_ranked": I[order].tolist(),
        "indices_topk": I[topk_idx].tolist(),
        "scores_topk": s[topk_idx].tolist()
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved ranking to {args.out} (topk={args.topk})")

if __name__ == "__main__":
    main()
