import os
import sys
import json
import shutil
import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
BASE_MODEL_NAME = "yolo26n-cls.pt"

DATA_ROOT = "dataset_cls_v2" if os.path.exists("dataset_cls_v2") else "dataset_cls"
WORK_DIR = os.path.abspath("runs_master_v12")

MANIFEST_PATH = os.path.join(DATA_ROOT, "manifest.json")
RAW_FRONT_DIR = os.path.join(DATA_ROOT, "raw_front")
RAW_FULL_DIR = os.path.join(DATA_ROOT, "raw_full")

TARGET_SIZE = 352
def _link_single_task(task):
    src, dest = task
    if not os.path.exists(dest):
        try:
            os.link(src, dest)
        except OSError:
            try:
                os.symlink(src, dest)
            except OSError:
                shutil.copy(src, dest)

def build_split_directories(items, split_name, base_folder, class_mapping, max_workers=32):
    """Fast parallel multi-threaded directory link generator."""
    unique_labels = set(class_mapping(x) for x in items)
    for lbl in unique_labels:
        os.makedirs(os.path.join(base_folder, split_name, str(lbl)), exist_ok=True)

    tasks = []
    for item in items:
        lbl = class_mapping(item)
        src = item["src"]
        dest = os.path.join(base_folder, split_name, str(lbl), f"{item['id']}.jpg")
        tasks.append((src, dest))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(_link_single_task, tasks),
            total=len(tasks),
            desc=f"Linking [{split_name}] ({len(tasks):,} items)",
            unit="img",
            leave=False
        ))

def get_class_idx(model, class_name):
    """Retrieves the output index for a specific class name safely."""
    for idx, name in model.names.items():
        if name == class_name:
            return idx
    raise ValueError(f"Class '{class_name}' not found in model classes: {model.names}")

def run_batched_predict(model, img_paths, target_class="1_car", batch_size=INFERENCE_BATCH_SIZE):
    """Vectorized high-speed GPU batched inference without per-sample CPU stalls."""
    target_idx = get_class_idx(model, target_class)
    probs = []
    
    with tqdm(total=len(img_paths), desc=f"Inference P({target_class})", unit="img", leave=False) as pbar:
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i + batch_size]
            results = model(batch, verbose=False, imgsz=TARGET_SIZE, device=DEVICE)
            batch_tensors = [r.probs.data for r in results if r.probs is not None]
            if len(batch_tensors) == len(batch):
                batch_probs = torch.stack(batch_tensors)[:, target_idx].detach().cpu().numpy()
                probs.extend(batch_probs.tolist())
            else:
                for r in results:
                    if r.probs is not None and len(r.probs.data) > target_idx:
                        probs.append(float(r.probs.data[target_idx].item()))
                    else:
                        probs.append(0.0)
            pbar.update(len(batch))
            
    return np.array(probs, dtype=np.float32)

def train_stage_model(model_name, stage_id, total_stages, dataset_dir, raw_dir, train_meta, val_meta, label_key, epochs=25):
    """Encapsulated helper for Stages 1, 2, and 4 with correct resume priorities."""
    run_dir = os.path.join(WORK_DIR, "runs", model_name)
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    print("\n" + "=" * 75)
    print(f"[STAGE {stage_id}/{total_stages}] Training {model_name} ({BASE_MODEL_NAME})")
    print("=" * 75)

    if os.path.exists(best_pt):
        print(f"[OK] Found completed weights for {model_name}: {best_pt}")
        print(f"[OK] SKIPPING Stage {stage_id} compute.")
        return best_pt

    if os.path.exists(last_pt):
        print(f"[OK] Found intermediate checkpoint for {model_name}: {last_pt}")
        print(f"[*] RESUMING Stage {stage_id} training...")
        model = YOLO(last_pt)
        model.train(resume=True)
        return best_pt

    t0 = time.time()
    shutil.rmtree(dataset_dir, ignore_errors=True)

    print(f"[*] Generating directory symlinks for {model_name}...")
    build_split_directories(
        [{"id": x["id"], "src": os.path.join(raw_dir, f"{x['id']}.jpg"), "lbl": x[label_key]} for x in train_meta],
        "train", dataset_dir, lambda x: x["lbl"]
    )
    build_split_directories(
        [{"id": x["id"], "src": os.path.join(raw_dir, f"{x['id']}.jpg"), "lbl": x[label_key]} for x in val_meta],
        "val", dataset_dir, lambda x: x["lbl"]
    )

    model = YOLO(BASE_MODEL_NAME)
    model.train(
        data=dataset_dir,
        epochs=epochs,
        patience=10,
        imgsz=TARGET_SIZE,
        device=DEVICE,
        batch=BATCH_SIZE,
        amp=True,
        cache=False,
        workers=NUM_WORKERS,
        cos_lr=True,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        project=os.path.join(WORK_DIR, "runs"),
        name=model_name,
        exist_ok=True
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def execute_pipeline():
    os.makedirs(WORK_DIR, exist_ok=True)
    log_file_path = os.path.join(WORK_DIR, "local_training_pipeline_v12.log")
    f_log = open(log_file_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, f_log)
    sys.stderr = Tee(sys.stderr, f_log)

    print("=" * 75)
    print("     BSM MASTER 5-STAGE TRI-CLASS PIPELINE (V12 UNIFIED RUN)")
    print(f"     Execution Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)
    print(f"[*] Base Architecture   : {BASE_MODEL_NAME}")
    print(f"[*] Dataset Root Path   : {os.path.abspath(DATA_ROOT)}")
    print(f"[*] Output Runs Path    : {WORK_DIR}")
    print(f"[*] Compute Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[*] Training Batch Size : {BATCH_SIZE} | Workers: {NUM_WORKERS}")
    print(f"[*] Temporal Downsample : Stride of {TRAIN_STRIDE} (20 FPS -> ~{20//TRAIN_STRIDE} FPS for Training)")
    print("=" * 75 + "\n")

    pipeline_start_t0 = time.time()
    if not os.path.exists(MANIFEST_PATH):
        print(f"[-] Error: Manifest not found at '{MANIFEST_PATH}'. Ensure dataset extraction has run.")
        sys.exit(1)

    print("[*] Loading manifest metadata...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    print(f"[OK] Loaded {len(manifest):,} cataloged frame crops.")

    print("[*] Grouping frames by continuous segment to prevent data leakage...")
    seg_groups = {}
    for item in manifest:
        k = (item["car_name"], item["route_id"], item["seg_idx"])
        seg_groups.setdefault(k, []).append(item)

    all_keys = list(seg_groups.keys())
    random.seed(42)
    random.shuffle(all_keys)
    n_val = max(1, int(len(all_keys) * 0.20))
    val_keys = set(all_keys[:n_val])
    train_meta = []
    val_meta = []

    for k, items in seg_groups.items():
        items_sorted = sorted(items, key=lambda x: str(x["id"]))
        if k in val_keys:
            val_meta.extend(items_sorted)
        else:
            train_meta.extend(items_sorted[::TRAIN_STRIDE])

    print(f"[OK] Segment Isolation & Sampling Complete:")
    print(f"    - Train Split : {len(train_meta):,} frames (Subsampled @ ~{20//TRAIN_STRIDE} FPS)")
    print(f"    - Val Split   : {len(val_meta):,} frames (Full 20 FPS temporal continuity retained)")
    print(f"    - Total Segs  : {len(all_keys)} segments ({len(all_keys) - n_val} Train / {n_val} Val)")
    best_a_path = train_stage_model(
        model_name="model_A",
        stage_id=1, total_stages=5,
        dataset_dir=os.path.join(WORK_DIR, "dataset_model_A"),
        raw_dir=RAW_FRONT_DIR,
        train_meta=train_meta, val_meta=val_meta,
        label_key="initial_label", epochs=25
    )
    best_b_path = train_stage_model(
        model_name="model_B",
        stage_id=2, total_stages=5,
        dataset_dir=os.path.join(WORK_DIR, "dataset_model_B"),
        raw_dir=RAW_FULL_DIR,
        train_meta=train_meta, val_meta=val_meta,
        label_key="initial_label", epochs=25
    )
    print("\n" + "=" * 75)
    print("[STAGE 3/5] Mining Spatial Discrepancies (V2 Probability Differential)")
    print("=" * 75)
    t0 = time.time()
    tri_manifest_path = os.path.join(WORK_DIR, "tri_class_manifest.json")
    
    if os.path.exists(tri_manifest_path):
        print(f"[OK] Found existing cached Tri-Class metadata at '{tri_manifest_path}'")
        with open(tri_manifest_path, 'r') as f:
            tri_class_meta = json.load(f)
        class_counts = {"0_nocar": 0, "1_car": 0, "2_distant_or_rear": 0}
        for item in tri_class_meta:
            class_counts[item["tri_label"]] += 1
    else:
        best_a = YOLO(best_a_path)
        best_b = YOLO(best_b_path)

        front_paths = [os.path.join(RAW_FRONT_DIR, f"{x['id']}.jpg") for x in manifest]
        full_paths = [os.path.join(RAW_FULL_DIR, f"{x['id']}.jpg") for x in manifest]

        print("[*] Running Dual-Model GPU Vector Inference across all frames...")
        probs_a = run_batched_predict(best_a, front_paths, target_class="1_car", batch_size=INFERENCE_BATCH_SIZE)
        probs_b = run_batched_predict(best_b, full_paths, target_class="1_car", batch_size=INFERENCE_BATCH_SIZE)

        tri_class_meta = []
        class_counts = {"0_nocar": 0, "1_car": 0, "2_distant_or_rear": 0}

        for idx, item in enumerate(manifest):
            pa = probs_a[idx]
            pb = probs_b[idx]
            if (pb >= 0.45 and pa < 0.30) or ((pb - pa) >= 0.25 and pb >= 0.40):
                final_label = "2_distant_or_rear"
            elif (pa >= 0.40) or (pb >= 0.65 and pa >= 0.35):
                final_label = "1_car"
            else:
                final_label = "0_nocar"

            class_counts[final_label] += 1
            item_copy = dict(item)
            item_copy["tri_label"] = final_label
            tri_class_meta.append(item_copy)

        with open(tri_manifest_path, 'w') as f:
            json.dump(tri_class_meta, f, indent=2)

    print(f"\n[+] Balanced Discrepancy Tri-Class Breakdown:")
    for k, v in class_counts.items():
        print(f"    - {k:20s}: {v:,} samples ({v/len(manifest)*100:.2f}%)")
    print(f"[OK] Stage 3 complete in {(time.time() - t0):.1f} seconds.")
    tri_seg_groups = {}
    for item in tri_class_meta:
        k = (item["car_name"], item["route_id"], item["seg_idx"])
        tri_seg_groups.setdefault(k, []).append(item)

    train_tri = []
    val_tri = []
    for k, items in tri_seg_groups.items():
        items_sorted = sorted(items, key=lambda x: str(x["id"]))
        if k in val_keys:
            val_tri.extend(items_sorted)
        else:
            train_tri.extend(items_sorted[::TRAIN_STRIDE])

    best_master_pt = train_stage_model(
        model_name="master_bsm",
        stage_id=4, total_stages=5,
        dataset_dir=os.path.join(WORK_DIR, "dataset_master_tri_class"),
        raw_dir=RAW_FULL_DIR,
        train_meta=train_tri, val_meta=val_tri,
        label_key="tri_label", epochs=50
    )

    onnx_path = best_master_pt.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        print("[*] Exporting Master Model weights to ONNX format...")
        final_onnx = YOLO(best_master_pt).export(format="onnx", imgsz=TARGET_SIZE, dynamic=False)
    else:
        print(f"[OK] Found existing exported ONNX model at: {onnx_path}")
        final_onnx = onnx_path
    print("\n" + "=" * 75)
    print("[STAGE 5/5] Vectorized Daemon Parameter & Hysteresis Sweep (20 FPS Time-Domain)")
    print("=" * 75)
    t0 = time.time()
    best_params, best_metrics = run_vectorized_daemon_sweep(val_tri, best_master_pt)
    print(f"[OK] Stage 5 complete in {(time.time() - t0):.1f} seconds.")
    print("\n" + "=" * 75)
    print("[*] Finalizing Deliverables in runs_master_v12...")
    print("=" * 75)
    deliverables_dir = os.path.join(WORK_DIR, "final_deliverables")
    os.makedirs(deliverables_dir, exist_ok=True)

    dest_onnx = os.path.join(deliverables_dir, "v_asm_model_master.onnx")
    shutil.copy(final_onnx, dest_onnx)
    shutil.copy(best_master_pt, os.path.join(deliverables_dir, "master_model_best.pt"))
    shutil.copy(best_a_path, os.path.join(deliverables_dir, "model_A_best.pt"))
    shutil.copy(best_b_path, os.path.join(deliverables_dir, "model_B_best.pt"))

    summary = {
        "version": "v12",
        "architecture": BASE_MODEL_NAME,
        "input_size": TARGET_SIZE,
        "train_downsample_stride": TRAIN_STRIDE,
        "best_optimal_parameters": best_params,
        "validation_metrics": best_metrics,
        "tri_class_distribution": class_counts
    }
    with open(os.path.join(deliverables_dir, "bsm_deployment_config.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    total_pipeline_time = (time.time() - pipeline_start_t0) / 60
    print("\n" + "=" * 75)
    print(f"[OK] ALL 5 PIPELINE STAGES COMPLETED IN {total_pipeline_time:.1f} MINUTES!")
    print(f"  - Master ONNX Model : {dest_onnx}")
    print(f"  - Deployment Config : {os.path.join(deliverables_dir, 'bsm_deployment_config.json')}")
    print(f"  - Deliverables Dir  : {deliverables_dir}")
    print("=" * 75 + "\n")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    f_log.close()

def run_vectorized_daemon_sweep(val_items, model_path):
    model = YOLO(model_path)
    val_items.sort(key=lambda x: (x["car_name"], x["route_id"], int(x["seg_idx"]), x.get("side", ""), x["id"]))

    img_paths = [os.path.join(RAW_FULL_DIR, f"{x['id']}.jpg") for x in val_items]
    raw_scores = run_batched_predict(model, img_paths, target_class="1_car", batch_size=INFERENCE_BATCH_SIZE)
    ground_truth = np.array([1 if x.get("initial_label", x.get("tri_label")) == "1_car" else 0 for x in val_items], dtype=np.int32)

    sequences = []
    curr_seq = [0]
    for i in range(1, len(val_items)):
        prev, curr = val_items[i-1], val_items[i]
        if (prev["car_name"] != curr["car_name"]) or (prev["route_id"] != curr["route_id"]) or (prev["seg_idx"] != curr["seg_idx"]) or (prev.get("side") != curr.get("side")):
            sequences.append(curr_seq)
            curr_seq = [i]
        else:
            curr_seq.append(i)
    sequences.append(curr_seq)

    CONF_TRIGGER_ON_GRID = [0.80, 0.85, 0.88, 0.90, 0.93, 0.95]
    CONF_HOLD_OFF_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    SMOOTHING_INTERVALS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    valid_configs = []
    for c_on in CONF_TRIGGER_ON_GRID:
        for c_off in CONF_HOLD_OFF_GRID:
            if c_off < c_on:
                for s_time in SMOOTHING_INTERVALS:
                    valid_configs.append((c_on, c_off, s_time))

    best_fitness = -999.0
    best_params = {}
    best_metrics = {}

    for c_on, c_off, s_time in tqdm(valid_configs, desc="Simulating Hysteresis Daemons", unit="cfg"):
        alpha = min(1.0, 0.05 / max(s_time, 0.001))
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_transitions = 0
        early_count = 0
        late_delays = []

        for seq_idx_list in sequences:
            seq_raw = raw_scores[seq_idx_list]
            seq_gt = ground_truth[seq_idx_list]

            active = False
            curr_ema = 0.0
            preds = np.zeros(len(seq_raw), dtype=np.int32)
            
            for t in range(len(seq_raw)):
                curr_ema = (1.0 - alpha) * curr_ema + alpha * seq_raw[t]
                if not active:
                    if curr_ema >= c_on:
                        active = True
                else:
                    if curr_ema < c_off:
                        active = False
                preds[t] = 1 if active else 0

            total_tp += np.sum((preds == 1) & (seq_gt == 1))
            total_fp += np.sum((preds == 1) & (seq_gt == 0))
            total_fn += np.sum((preds == 0) & (seq_gt == 1))
            total_transitions += np.sum(np.diff(preds) > 0)
            gt_padded = np.pad(seq_gt, (1, 0), constant_values=0)
            pred_padded = np.pad(preds, (1, 0), constant_values=0)
            gt_onsets = np.where(np.diff(gt_padded) > 0)[0]
            pred_onsets = np.where(np.diff(pred_padded) > 0)[0]

            for gt_on in gt_onsets:
                cands = pred_onsets[pred_onsets >= gt_on - 10]
                if len(cands) > 0:
                    if cands[0] < gt_on:
                        early_count += 1
                    else:
                        late_delays.append(cands[0] - gt_on)
                else:
                    late_delays.append(50)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        beta = 0.5
        fbeta = ((1 + beta**2) * prec * rec) / ((beta**2 * prec) + rec) if (prec + rec) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        fp_penalty = (total_fp / max(total_tp + total_fp, 1)) * 0.20
        latency_penalty = np.mean(late_delays) * 0.2 * 0.01 if late_delays else 0.0
        early_bonus = min(early_count * 0.001, 0.05)
        fitness = fbeta - fp_penalty - latency_penalty + early_bonus

        if fitness > best_fitness:
            best_fitness = fitness
            best_params = {
                "conf_trigger_on": c_on,
                "conf_hold_off": c_off,
                "smooth_sec": s_time
            }
            best_metrics = {
                "F0.5": float(fbeta),
                "F1": float(f1),
                "Precision": float(prec),
                "Recall": float(rec),
                "FP": int(total_fp),
                "FN": int(total_fn),
                "EarlyDet": int(early_count),
                "LateMissed": len(late_delays),
                "Transitions": int(total_transitions),
                "Composite_Fitness": float(fitness)
            }

    m = best_metrics
    print("\n" + "=" * 75)
    print(f"  [{BASE_MODEL_NAME}] - Optimal Production Configuration Discovered:")
    print("=" * 75)
    print(f"    Trigger ON Threshold      : {best_params['conf_trigger_on']*100:.1f}%")
    print(f"    Hold OFF Threshold        : {best_params['conf_hold_off']*100:.1f}%")
    print(f"    Temporal Smoothing        : {best_params['smooth_sec']:.2f} s")
    print("    " + "-"*55)
    print(f"    F0.5 (precision-weighted) : {m['F0.5'] * 100:.2f}%")
    print(f"    F1 (balanced)             : {m['F1'] * 100:.2f}%")
    print(f"    Precision                 : {m['Precision'] * 100:.2f}%")
    print(f"    Recall                    : {m['Recall'] * 100:.2f}%")
    print(f"    False Positives           : {m['FP']:,}")
    print(f"    False Negatives           : {m['FN']:,}")
    print(f"    Early Detections (good)   : {m['EarlyDet']:,}")
    print(f"    Late/Missed (bad)         : {m['LateMissed']:,}")
    print(f"    Oscillations (state chg)  : {m['Transitions']:,}")
    print(f"    Composite Fitness         : {best_fitness:.4f}\n")

    return best_params, best_metrics


if __name__ == "__main__":
    execute_pipeline()
