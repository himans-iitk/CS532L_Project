# Continuously Changing Corruptions (CCC)

Code to run test-time adaptation experiments on the CCC benchmark. This builds on the NeurIPS 2023 paper “RDumb: A simple approach that questions our progress in continual test-time adaptation” (https://arxiv.org/pdf/2306.05401) and the upstream repo (https://github.com/oripress/CCC). CCC streams ImageNet-derived corruptions online; no local dataset generation is required.

## What’s here
- `CCC/eval.py`: evaluation entry point
- `CCC/models/`: pretrained, rdumb, rdumbpp variants, tent, eata, etc.
- `CCC/models/rdumbpp.py`: RDumb++ implementations (entropy- and KL-based, full/soft reset) with drift detection and soft resets over time.
- `CCC_Evaluation.ipynb`: end-to-end runs and ablations

## Quick start (CUDA machine / RunPod)
1) Clone and enter:
   ```bash
   git clone https://github.com/himans-iitk/Rdumbpp.git
   cd Rdumbpp
   ```
2) Install deps (CUDA PyTorch 2.0.1 + torchvision 0.15.2 recommended):
   ```bash
   pip install "numpy<2.0" torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
   pip install webdataset Pillow
   ```
3) Run an evaluation (example: RDumb on CCC-medium, all 3 seeds × 3 speeds):
   ```bash
   for i in $(seq 0 8); do
     python eval.py --mode rdumb --baseline 20 --logs logs --processind $i
   done
   ```
   Outputs land in `logs/ccc_20/` as text files (per-batch accuracies).

## Models (registry names)
- `pretrained` (baseline, no adaptation)
- `rdumb`
- `rdumbpp_ent_full`, `rdumbpp_ent_soft`, `rdumbpp_kl_full`, `rdumbpp_kl_soft`
- Other baselines: `tent`, `eta`, `eata`, `cotta`, etc.

## Dataset streaming
CCC shards are streamed; no generation needed:
```
https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/baseline_{baseline}_transition+speed_{speed}_seed_{seed}/serial_{00000..99999}.tar
```
Baselines: 0, 20, 40. Speeds: 1000, 2000, 5000. Seeds: 43, 44, 45.

## Notebook workflow (`CCC_Evaluation.ipynb`)
- Configures baseline, seeds, speeds, model list.
- Runs all models over ~1M images per seed (3 seeds × 3 speeds).
- RDumb++ ablations:
  - k ∈ {2.0, 2.5, 3.0} (EntropyFull @ speed 2000)
  - λ ∈ {0.30, 0.50, 0.70} (EntropySoft @ speed 5000, k=2.5)
- Aggregates tables and writes CSVs.

## Notes
- Use a CUDA machine; CPU is extremely slow.
- `eval.py` caps runs at ~1M images; adjust if you need full 7.5M.
- Model registry is decorator-based; add via `@register(<name>)` and import in `models/__init__.py`.

## Attribution
Built on CCC and RDumb from the NeurIPS 2023 paper (https://arxiv.org/pdf/2306.05401) and upstream repo (https://github.com/oripress/CCC).
