import os
import sys
import json
import time
import shutil
import random
import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEOS_DIR = "videos"
MASK_REPO_FULL = "mask_repo"
MASK_REPO_FRONT = "mask_repofront"

OUTPUT_DATASET_DIR = "dataset_cls_v2"
TARGET_SQUARE_SIZE = 352

NUM_WORKERS = 8

HARD_NEG_PRE_SEC = 1.0
HARD_NEG_POST_SEC = 1.0

def letterbox_square(img, target_size=TARGET_SQUARE_SIZE):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    scale = target_size / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=img.dtype)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def load_mask_geometry(mask_path):
    if not os.path.exists(mask_path):
        return None
    try:
        with open(mask_path, "r") as f:
            data = json.load(f)
        poly_l = data.get("poly_left", [])
        poly_r = data.get("poly_right", [])
        if len(poly_l) < 3 or len(poly_r) < 3:
            return None
        pl = np.array(poly_l, dtype=np.int32)
        lx, ly, lw, lh = cv2.boundingRect(pl)
        mask_l = np.zeros((lh, lw), dtype=np.uint8)
        cv2.fillPoly(mask_l, [pl - [lx, ly]], 255)
        pr = np.array(poly_r, dtype=np.int32)
        rx, ry, rw, rh = cv2.boundingRect(pr)
        mask_r = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask_r, [pr - [rx, ry]], 255)
        return {"lx": lx, "ly": ly, "lw": lw, "lh": lh, "mask_l": mask_l, "rx": rx, "ry": ry, "rw": rw, "rh": rh, "mask_r": mask_r}
    except Exception:
        return None

def process_single_segment_task(task):
    """Worker task wrapper for multiprocessing."""
    seg_idx, route_id, route_dir, labels, geom_full, geom_front, car_name = task
    extracted_meta = []
    
    mp4_path = os.path.join(route_dir, f"{seg_idx}_dcamera.mp4")
    hevc_path = os.path.join(route_dir, f"{seg_idx}_dcamera.hevc")
    video_path = mp4_path if os.path.exists(mp4_path) else hevc_path
    if not os.path.exists(video_path):
        return extracted_meta

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1200
    safe_name = f"r_{route_id}_s_{seg_idx}"

    left_ivs = labels.get("left", [])
    right_ivs = labels.get("right", [])
    sample_plan = {}
    for f in range(total_frames):
        t = f / fps
        is_pos_l = any(it["start"] <= t <= it["end"] for it in left_ivs)
        is_pos_r = any(it["start"] <= t <= it["end"] for it in right_ivs)

        is_hard_neg_l = (not is_pos_l) and any((it["start"] - HARD_NEG_PRE_SEC) <= t <= (it["end"] + HARD_NEG_POST_SEC) for it in left_ivs)
        is_hard_neg_r = (not is_pos_r) and any((it["start"] - HARD_NEG_PRE_SEC) <= t <= (it["end"] + HARD_NEG_POST_SEC) for it in right_ivs)

        need_l = False
        need_r = False
        type_l = "0_nocar"
        type_r = "0_nocar"

        if is_pos_l:
            if f % 4 == 0:
                need_l = True
                type_l = "1_car"
        elif is_hard_neg_l:
            if f % 6 == 0:
                need_l = True
                type_l = "0_nocar"
        else:
            if f % 30 == 0:
                need_l = True
                type_l = "0_nocar"

        if is_pos_r:
            if f % 4 == 0:
                need_r = True
                type_r = "1_car"
        elif is_hard_neg_r:
            if f % 6 == 0:
                need_r = True
                type_r = "0_nocar"
        else:
            if f % 30 == 0:
                need_r = True
                type_r = "0_nocar"

        if need_l or need_r:
            sample_plan[f] = (need_l, type_l, need_r, type_r)

    frame_count = 0
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 93]

    while True:
        if frame_count in sample_plan:
            ret, frame = cap.read()
            if not ret: break
            vh, vw = frame.shape[:2]
            need_l, type_l, need_r, type_r = sample_plan[frame_count]
            if need_l:
                g = geom_front
                crop = frame[g["ly"]:min(g["ly"]+g["lh"], vh), g["lx"]:min(g["lx"]+g["lw"], vw)]
                masked = cv2.bitwise_and(crop, crop, mask=g["mask_l"][:crop.shape[0], :crop.shape[1]])
                img_front = letterbox_square(masked)
                g = geom_full
                crop = frame[g["ly"]:min(g["ly"]+g["lh"], vh), g["lx"]:min(g["lx"]+g["lw"], vw)]
                masked = cv2.bitwise_and(crop, crop, mask=g["mask_l"][:crop.shape[0], :crop.shape[1]])
                img_full = letterbox_square(masked)

                base_id = f"{safe_name}_f{frame_count:05d}_L"
                path_front = os.path.join(OUTPUT_DATASET_DIR, "raw_front", f"{base_id}.jpg")
                path_full = os.path.join(OUTPUT_DATASET_DIR, "raw_full", f"{base_id}.jpg")

                cv2.imwrite(path_front, img_front, jpeg_params)
                cv2.imwrite(path_full, img_full, jpeg_params)

                extracted_meta.append({
                    "id": base_id,
                    "car_name": car_name,
                    "route_id": route_id,
                    "seg_idx": seg_idx,
                    "side": "left",
                    "initial_label": type_l,
                    "path_front": path_front,
                    "path_full": path_full
                })
            if need_r:
                g = geom_front
                crop = frame[g["ry"]:min(g["ry"]+g["rh"], vh), g["rx"]:min(g["rx"]+g["rw"], vw)]
                masked = cv2.bitwise_and(crop, crop, mask=g["mask_r"][:crop.shape[0], :crop.shape[1]])
                img_front = letterbox_square(masked)

                g = geom_full
                crop = frame[g["ry"]:min(g["ry"]+g["rh"], vh), g["rx"]:min(g["rx"]+g["rw"], vw)]
                masked = cv2.bitwise_and(crop, crop, mask=g["mask_r"][:crop.shape[0], :crop.shape[1]])
                img_full = letterbox_square(masked)

                base_id = f"{safe_name}_f{frame_count:05d}_R"
                path_front = os.path.join(OUTPUT_DATASET_DIR, "raw_front", f"{base_id}.jpg")
                path_full = os.path.join(OUTPUT_DATASET_DIR, "raw_full", f"{base_id}.jpg")

                cv2.imwrite(path_front, img_front, jpeg_params)
                cv2.imwrite(path_full, img_full, jpeg_params)

                extracted_meta.append({
                    "id": base_id,
                    "car_name": car_name,
                    "route_id": route_id,
                    "seg_idx": seg_idx,
                    "side": "right",
                    "initial_label": type_r,
                    "path_front": path_front,
                    "path_full": path_full
                })
        else:
            if not cap.grab(): break

        frame_count += 1

    cap.release()
    return extracted_meta

def main():
    print("\n" + "="*65)
    print("   BSM DUAL-MASK EXTRACTION PIPELINE (dataset_cls_v2)")
    print(f"   [Multiprocessing Enabled: {NUM_WORKERS} Workers]")
    print("="*65 + "\n")

    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "raw_front"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET_DIR, "raw_full"), exist_ok=True)

    all_car_routes = []
    if os.path.exists(VIDEOS_DIR):
        for car_name in os.listdir(VIDEOS_DIR):
            car_dir = os.path.join(VIDEOS_DIR, car_name)
            if not os.path.isdir(car_dir): continue
            for route_id in sorted(os.listdir(car_dir)):
                route_dir = os.path.join(car_dir, route_id)
                config_path = os.path.join(route_dir, "annotations.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            r_config = json.load(f)
                        if r_config.get("is_finished", False):
                            all_car_routes.append((car_name, route_id, r_config))
                    except Exception: pass

    print(f"[*] Found {len(all_car_routes)} completed annotated routes.")
    tasks = []
    mask_cache = {}

    for car_name, route_id, r_config in all_car_routes:
        if car_name not in mask_cache:
            mask_file = f"{car_name}.json"
            full_p = os.path.join(MASK_REPO_FULL, mask_file)
            front_p = os.path.join(MASK_REPO_FRONT, mask_file)
            geom_full = load_mask_geometry(full_p)
            geom_front = load_mask_geometry(front_p)
            mask_cache[car_name] = (geom_full, geom_front)
        
        geom_full, geom_front = mask_cache[car_name]
        if not geom_full or not geom_front:
            continue

        route_dir = os.path.join(VIDEOS_DIR, car_name, route_id)
        completed_segs = sorted(list(set([
            f.split('_')[0] for f in os.listdir(route_dir)
            if (f.endswith('_dcamera.hevc') or f.endswith('_dcamera.mp4')) and f.split('_')[0].isdigit()
        ])), key=int)

        for seg_idx in completed_segs:
            labels = r_config.get("segments", {}).get(seg_idx, {"left": [], "right": []})
            tasks.append((seg_idx, route_id, route_dir, labels, geom_full, geom_front, car_name))

    print(f"[*] Total video segments queued: {len(tasks)}")

    master_meta = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_segment_task, task): task for task in tasks}
        with tqdm(total=len(tasks), desc="Processing Segments") as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    master_meta.extend(res)
                except Exception as e:
                    print(f"\n[!] Error processing segment: {e}")
                pbar.update(1)

    elapsed = time.time() - start_time
    print(f"\n[+] Total Frames Extracted: {len(master_meta)} in {elapsed:.1f}s ({len(tasks)/max(elapsed, 1):.2f} segments/sec)")
    meta_path = os.path.join(OUTPUT_DATASET_DIR, "manifest.json")
    with open(meta_path, 'w') as f:
        json.dump(master_meta, f, indent=2)
    zip_name = "dataset_cls_v2.zip"
    print(f"\n[*] Zipping {OUTPUT_DATASET_DIR} into '{zip_name}'...")
    zip_start = time.time()
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(OUTPUT_DATASET_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUTPUT_DATASET_DIR)
                zipf.write(file_path, arcname)

    print(f"[OK] Zip created in {time.time() - zip_start:.1f}s!")
    print(f"[OK] DONE! Upload '{zip_name}' directly to your Kaggle Notebook Dataset.")

if __name__ == "__main__":
    main()
