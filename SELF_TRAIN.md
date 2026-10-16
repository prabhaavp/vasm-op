## 1. Install dependencies

Use Python 3.10+ and install PyTorch for your hardware, then:

```powershell
python -m pip install ultralytics numpy tqdm opencv-python onnx onnxruntime
```

The scripts use `torch` through PyTorch. The base classifier is `yolo26n-cls.pt`; place it in the project folder or allow Ultralytics to download it.

## 2. Prepare data

For automatic video-to-crop extraction, provide annotated routes and paired masks using this layout:

```text
videos/<vehicle>/<route>/annotations.json
videos/<vehicle>/<route>/<segment>_dcamera.mp4   # or .hevc
mask_repo/<vehicle>.json
mask_repofront/<vehicle>.json
```

Each mask JSON needs `poly_left` and `poly_right`. Each completed route annotation needs segment `left` / `right` event intervals with `start` and `end` seconds. Then run:

```powershell
python prepare_dataset.py
```

This writes `dataset_cls_v2/manifest.json` and paired images in `raw_front/` and `raw_full/`. If you already have prepared crops, skip extraction and place them in that layout. The manifest must include `id`, `car_name`, `route_id`, `seg_idx`, and `initial_label` (`0_nocar` or `1_car`).

## 3. Train from the base model

From the directory containing the scripts and `dataset_cls_v2`, run:

```powershell
python train_pipeline.py
```

Outputs are saved under `runs_master_v12/`; final weights and ONNX are in `runs_master_v12/final_deliverables/`. The pipeline first trains front/full classifiers, mines `distant_or_rear` pseudo-labels from their score differences, trains the three-class model, then tunes temporal alarm settings.

## Fine-tune or adapt

Edit the constants at the top of `train_pipeline.py` before running:

- `BASE_MODEL_NAME`: checkpoint to train from.
- `DATA_ROOT`, `WORK_DIR`: dataset and output paths.
- `TARGET_SIZE`, `BATCH_SIZE`, `NUM_WORKERS`: image and hardware settings.
- `TRAIN_STRIDE`: training frame subsampling.

Change `epochs` and augmentation options in the `model.train(...)` call. To change pseudo-label behavior, edit the Stage 3 thresholds in `execute_pipeline()`. To tune alarm behavior, edit the threshold and smoothing grids in `run_vectorized_daemon_sweep()`.

For a clean retrain, remove the stage's saved run folder under `runs_master_v12/runs/` (otherwise an existing `best.pt` makes the stage skip). To rebuild pseudo-labels after changing Stage 3, also remove `runs_master_v12/tri_class_manifest.json`. Keep validation grouped by `(car_name, route_id, seg_idx)` so adjacent frames from one segment do not appear in both train and validation. Review mined labels and re-sweep temporal thresholds on your own validation routes.

## Why the pipeline is structured this way

Video frames are highly correlated, so validation is split by route segment and training frames are subsampled. Separate front and full views let the system identify distant/rear examples from model disagreement, reducing manual labeling but introducing pseudo-label noise. Hard links are tried before copies to limit disk use, with copy fallback for filesystems that do not support links. Batch size and worker settings may need to be adjusted on other hardware. EMA smoothing with separate on/off thresholds reduces alarm chatter; its best values depend on deployment data.
## Anonymized sample

`examples/sample_vehicle/` contains an anonymized sample route, its segment video, `annotations.json`, paired masks, and illustrative manifest records. The original device and route IDs have been replaced with synthetic values. To run extraction on this sample, open a terminal in `examples/sample_vehicle` and run `python ..\..\prepare_dataset.py`; the script expects `videos/Example Vehicle/sample_route_7c91e4a2/13_dcamera.mp4` and `annotations.json` in that route folder, plus `mask_repo/Example Vehicle.json` and `mask_repofront/Example Vehicle.json`. This creates sample crops and replaces `dataset_cls_v2/manifest.json`, so back up the illustrative manifests first if you want to keep them.

The annotation labels the right side of segment 13 from 22.6 to 24.9 seconds. Mask vertices use the video pixel coordinates (1928 x 1208); the full and front masks intentionally differ. Relevant video segments and time intervals were identified with a semi-automated workflow and reviewed manually. For new data, a small hotkey-driven video viewer is an easy way to review frames: step through the clip, mark interval start and end points, and inspect the regions of interest. The sample polygons match this camera view. Create or adjust masks for your camera, then review the extracted frames and labels before training.