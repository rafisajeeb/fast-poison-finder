# Fast Poison-Point Finder (CIFAR-10)

A lightweight, reproducible pipeline to **create poisoned CIFAR-10 splits**, **train a baseline/backdoored model**, **embed the training set with a frozen ResNet-18**, **rank suspicious samples** (Mahalanobis + LOF fusion), and **evaluate Precision@k**.

> Designed to run on a modest machine with Python + PyTorch. CIFAR-10 auto-downloads.

## Quick Start

```bash
# (1) Create a Python env and install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (2) Make a poisoned index file (10% blended trigger -> target class 0)
python scripts/make_poison_indices.py --poison-rate 0.10 --target-class 0 --seed 1 --out data/poisoned_blended_p10_s1.json

# (3) Train a model on clean OR poisoned data (pass --poison-indices to poison the training set)
python scripts/train_model.py --epochs 40 --save ckpt/clean_resnet18.pth
python scripts/train_model.py --epochs 40 --poison-indices data/poisoned_blended_p10_s1.json --save ckpt/poisoned_resnet18.pth

# (4) Measure clean accuracy and attack success rate (ASR)
python scripts/eval_asr.py --model ckpt/poisoned_resnet18.pth --poison-settings target=0,size=5,alpha=0.2

# (5) Extract frozen-backbone embeddings for the (poisoned) training set
python scripts/compute_embeddings.py --poison-indices data/poisoned_blended_p10_s1.json --out data/embeddings_blended_p10_s1.npz

# (6) Rank suspicious samples (Mahalanobis, LOF, and Fusion) and save Top-k list
python scripts/rank_suspicious.py --embeddings data/embeddings_blended_p10_s1.npz --method fusion --topk 200 --out data/rank_fusion_top200.json

# (7) Evaluate Precision@k against ground-truth poisoned indices
python scripts/eval_precision.py --rank data/rank_fusion_top200.json --poison-indices data/poisoned_blended_p10_s1.json --k 200
```

## Files

- `scripts/make_poison_indices.py`: picks which training samples will be poisoned (saved as JSON indices).
- `scripts/train_model.py`: trains ResNet-18 on clean or poisoned data (applies blended square trigger online).
- `scripts/eval_asr.py`: evaluates **clean accuracy** and **attack success rate (ASR)**.
- `scripts/compute_embeddings.py`: saves cached embeddings/labels/indices with a frozen ImageNet ResNet-18.
- `scripts/rank_suspicious.py`: ranks samples via **Mahalanobis**, **LOF**, or **Fusion**; outputs Top-k JSON.
- `scripts/eval_precision.py`: computes **Precision@k** given ground-truth poisoned indices.
- `src/poison.py`: trigger functions (blended white square patch) and helpers.
- `src/utils.py`: utilities (seeding, device, dataset wrappers).
- `requirements.txt`: Python dependencies.

## Notes

- CIFAR-10 auto-downloads to `~/.torch` or `./data` depending on your PyTorch config.
- Default trigger: a small white square at bottom-right, blended with alpha=0.2, target class `0` (airplane).
- For label-consistent or WaNet-style attacks, extend `src/poison.py` and the CLI in `make_poison_indices.py` and `train_model.py`.

## Reproducibility

- Fixed seeds (`--seed`), deterministic flags where possible.
- All scripts print and save JSON logs (`--save-json` planned for extensions).

---

**Next steps**: after you get Precision@k and ASR numbers, try different poison rates (5%, 10%), trigger sizes, and LOF neighbors (10–30), then plot P@k curves.
