# DL_baselines integration layer

This folder integrates five deep-learning baselines into the ForensicFusion project **without polluting the main environment**.

Supported baselines:
- TruFor (official repo)
- MMFusion (official repo)
- OMG-Fuser (official repo; signal-driven / CSV-driven)
- ManTraNet (official repo; notebook-derived inference runner)
- CAT-Net (official repo; patched runtime wrapper that saves numeric maps)

## Design goals

- keep the main `forensicfusion/` codebase clean;
- run official repos in **isolated conda environments** whenever possible;
- save predictions in the same layout already expected by `forensicfusion.external.ExternalPredictionStore`:

```text
predictions/
  <DATASET>/
    <METHOD>/<sample_id>.png
    <METHOD>_scores.csv
```

## Quick start

Clone the repos:

```bash
python scripts/setup_dl_baselines.py --methods TruFor,MMFusion,ManTraNet,CAT-Net,OMG-Fuser --clone_only
```

Create isolated envs (optional but recommended):

```bash
python scripts/setup_dl_baselines.py --methods TruFor,MMFusion,ManTraNet,CAT-Net,OMG-Fuser --create_envs
```

Run baselines on a dataset split:

```bash
python scripts/run_dl_baselines.py   --dataset CASIA_v1   --data_root data   --split test   --methods TruFor,MMFusion,ManTraNet,CAT-Net   --predictions_root predictions
```

Then evaluate them together with ForensicFusion using the existing table script:

```bash
python scripts/table1_quantitative.py   --datasets CASIA_v1   --data_root data   --cache_root cache   --gnn_ckpt runs/gnn/ckpt.pt   --fusion_ckpt runs/fusion/fusion_weights.pt   --external_pred_root predictions   --external_methods TruFor,MMFusion,ManTraNet,CAT-Net
```

## Notes by method

### TruFor
- The official repo ships a test script that accepts a file/folder/glob and saves `.npz` files containing `map`, `score`, `conf`, and optionally `np++`.
- The adapter can auto-download the official inference weights zip if you pass `--trufor_auto_download_weights`.

### MMFusion
- The official repo provides `inference.py` for single-image inference.
- The adapter runs it image-by-image and parses the printed detection score.

### ManTraNet
- The official repo mainly exposes inference through the demo notebook.
- The adapter uses the **same official preprocessing and model loading** shown in that notebook and runs the Keras/TensorFlow model directly over a directory of images.

### CAT-Net
- The official `tools/infer.py` hardcodes both model selection and plotting.
- The adapter patches it at runtime so it saves **numeric grayscale maps** instead of colored heatmaps, which makes quantitative evaluation possible.

### OMG-Fuser
- The official repo expects a dataset CSV and precomputed signal files rather than raw-image-only inference.
- For that reason, the adapter is intentionally **CSV-driven**. Use `scripts/prepare_omgfuser_csv.py` to build a CSV from already prepared signals.
