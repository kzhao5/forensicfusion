# ForensicFusion (practical project scaffold)

This repository is a **practical implementation scaffold** for the ForensicFusion idea in the workshop draft:

- build a **forensic supernet** from pyIFD-style modules,
- sample candidate **forensic analysis paths**,
- train a **GraphSAGE-style path selector**,
- optionally train **manipulation-type-conditioned fusion weights**,
- run the method on images and export **CVPR-friendly experiment tables**.

The code is designed to be runnable even when only part of pyIFD is available:

- the **core modules** (`ELA`, `DCT`, `NOI1`, `NOI4`, `GHOST`, `BLK`) have portable local fallbacks;
- the **optional modules** (`NOI2`, `NOI5`, `ADQ*`, `CFA1`, `CAGI`, `NADQ`) are used when pyIFD is installed successfully.

## Why this version is more practical than the abstract paper draft

The draft describes a generic supernet/DAG with GNN- or LLM-guided path selection. In code, a more stable formulation is:

1. **Path = subset of parallel heatmap-producing modules + fusion rule**.
2. **LLM is optional**, not required for the core experiments.
3. **Selection is image-conditioned but deterministic** through a trainable GraphSAGE-style selector.
4. **Fusion is lightweight and trainable** with per-manipulation-type weights.
5. The experiment scripts directly generate the tables you need for:
   - quantitative comparison,
   - selection/fusion ablation,
   - efficiency trade-off,
   - top-k sensitivity,
   - type robustness,
   - cross-dataset evaluation,
   - qualitative figures.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

Install pyIFD from GitHub:

```bash
pip install git+https://github.com/EldritchJS/pyIFD
```

If some optional pyIFD dependencies fail, the project still runs with the core fallback modules.

## Dataset format

For each dataset create:

```text
data/<DATASET_NAME>/
  images/
    <id>.jpg|png|...
  masks/
    <id>.png            # optional for authentic images / detection-only images
  meta.csv
```

`meta.csv` must contain:

- `id`
- `split` in `{train,val,test}`
- `manip_type` in `{splicing,copy_move,inpainting,ai_generated,unknown}`

Optional:

- `label` in `{0,1}` where `1=manipulated`, `0=authentic`

If `label` is missing, the loader infers `label=1` when a mask exists, otherwise `0`.

Example:

```csv
id,split,manip_type,label
000001,train,splicing,1
000002,val,copy_move,1
000003,test,unknown,0
```

## Step 1: precompute module outputs

```bash
python scripts/precompute_outputs.py \
  --dataset CASIA_v1 \
  --data_root data \
  --out_root cache \
  --split all
```

To use the larger pyIFD set:

```bash
python scripts/precompute_outputs.py \
  --dataset CASIA_v1 \
  --data_root data \
  --out_root cache \
  --split all \
  --include_optional
```

## Step 2: train the GNN path selector

```bash
python scripts/train_gnn.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --out_dir runs/gnn \
  --K 50 \
  --min_len 1 \
  --max_len 4 \
  --epochs 15
```

## Step 3: train learned fusion weights

```bash
python scripts/train_fusion.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --out_dir runs/fusion \
  --epochs 10
```

## Step 4: run the key experiments

### Table 1: quantitative comparison (ours + pyIFD baselines + external methods)

```bash
python scripts/table1_quantitative.py \
  --datasets CASIA_v1,Coverage,Columbia,DSO_1,CocoGlide \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt \
  --external_pred_root predictions \
  --external_methods TruFor,MMFusion
```

External methods are read from saved prediction maps, for example:

```text
predictions/
  CASIA_v1/
    TruFor/000001.png
    MMFusion/000001.png
```

### Table 2: selection + fusion ablation

```bash
python scripts/ablation_selection_fusion.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt
```

### Table 3: efficiency trade-off

```bash
python scripts/efficiency_tradeoff.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt \
  --Ks 5,10,20,50,100
```

For timing without cache:

```bash
python scripts/efficiency_tradeoff.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt \
  --time_without_cache
```

## Additional experiments

### top-k sensitivity

```bash
python scripts/topk_sensitivity.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt
```

### robustness to wrong manipulation-type input

```bash
python scripts/type_robustness.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt
```

### cross-dataset generalization

```bash
python scripts/cross_dataset_generalization.py \
  --train_name CASIA_v1 \
  --test_datasets Coverage,Columbia,DSO_1,CocoGlide \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt
```

### qualitative grids

```bash
python scripts/qualitative_grid.py \
  --dataset CASIA_v1 \
  --data_root data \
  --cache_root cache \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt \
  --external_pred_root predictions \
  --external_methods TruFor,MMFusion
```

## Single-image demo

```bash
python scripts/run_on_image.py \
  --image /path/to/test.jpg \
  --manip_type splicing \
  --gnn_ckpt runs/gnn/ckpt.pt \
  --fusion_ckpt runs/fusion/fusion_weights.pt \
  --out_dir demo_out
```

This writes:

- raw heatmaps,
- overlays,
- a qualitative grid,
- the ranked selected paths.

## Notes on MeVer / website results

The code does **not** depend on calling a remote website. Instead, it supports reading any externally generated heatmaps from disk and scoring them consistently through `scripts/eval_external_predictions.py` and `scripts/table1_quantitative.py`.

That is the safest way to incorporate:

- MeVer result screenshots converted to maps,
- outputs from TruFor/MMFusion/other repos,
- internal baselines you already ran elsewhere.
