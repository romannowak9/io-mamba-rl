import os
import json
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

# ==========================================
# FUNKCJE POMOCNICZE (IoU I PARSERY)
# ==========================================

def calculate_box_iou(box1, box2):
    """Oblicza IoU dla dwóch boksów w formacie MOT [tl_x, tl_y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Konwersja do tlbr (top-left, bottom-right)
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2

    # Współrzędne przecięcia
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Pola powierzchni
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def load_afo_gt_mot(json_path):
    """Wczytuje Ground Truth z JSON i zwraca listę boksów w formacie MOT [tl_x, tl_y, w, h]"""
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    gt_boxes = []
    for obj in data.get("objects", []):
        if obj["geometryType"] == "rectangle" and obj["classTitle"] == "object":
            pts = obj["points"]["exterior"]
            if len(pts) == 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                tl_x = min(x1, x2)
                tl_y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                gt_boxes.append([tl_x, tl_y, w, h])
    return gt_boxes

def match_detections_to_gt(det_boxes, gt_boxes, iou_threshold=0.5):
    """
    Zachłanne dopasowanie detekcji do Ground Truth na podstawie macierzy IoU.
    Zwraca liczbę (TP, FP, FN) oraz listę IoU dla dopasowanych boksów.
    """
    if len(det_boxes) == 0:
        return 0, 0, len(gt_boxes), []
    if len(gt_boxes) == 0:
        return 0, len(det_boxes), 0, []

    # Budowanie macierzy IoU
    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
    for d_idx, det in enumerate(det_boxes):
        for g_idx, gt in enumerate(gt_boxes):
            iou_matrix[d_idx, g_idx] = calculate_box_iou(det, gt)

    tp, fp = 0, 0
    matched_gts = set()
    matched_ious = []

    # Sortowanie indeksów detekcji od najwyższych wartości IoU
    sorted_det_indices = np.argsort(-np.max(iou_matrix, axis=1))

    for d_idx in sorted_det_indices:
        best_g_idx = np.argmax(iou_matrix[d_idx])
        best_iou = iou_matrix[d_idx, best_g_idx]

        if best_iou >= iou_threshold and best_g_idx not in matched_gts:
            tp += 1
            matched_gts.add(best_g_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gts)
    return tp, fp, fn, matched_ious

# ==========================================
# GŁÓWNA PĘTLA EWALUACYJNA
# ==========================================

def evaluate_detector(data_root, det_results_dir, split="train", conf_thresh=0.5, iou_thresh=0.5):
    split_map = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
    target_split = split_map[split]
    
    img_dir = Path(data_root) / target_split / "img"
    ann_dir = Path(data_root) / target_split / "ann"
    
    # Krok 1: Odtworzenie grupowania sekwencji (dokładnie tak jak w gen_det_afo.py)
    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    seq_groups = {}
    for img_path in all_images:
        stem = img_path.stem
        parts = stem.split("_")
        if len(parts) < 2: continue
        seq_id = "_".join(parts[:-1])
        try:
            frame_idx = int(parts[-1])
        except ValueError: continue
            
        if seq_id not in seq_groups:
            seq_groups[seq_id] = []
        seq_groups[seq_id].append((frame_idx, img_path))

    global_tp, global_fp, global_fn = 0, 0, 0
    global_matched_ious = []
    
    table_data = []

    # Krok 2: Ewaluacja sekwencja po sekwencji
    for seq_id, frame_list in seq_groups.items():
        det_txt_path = os.path.join(det_results_dir, f"{seq_id}.txt")
        if not os.path.exists(det_txt_path):
            print(f"Brak pliku detekcji dla sekwencji: {det_txt_path}. Pomijam.")
            continue
            
        # Sortowanie chronologiczne klatek
        frame_list.sort(key=lambda x: x[0])
        
        # Wczytanie wygenerowanych detekcji [.txt]
        try:
            all_dets = np.loadtxt(det_txt_path, delimiter=',')
        except Exception:
            all_dets = np.empty((0, 10))
            
        if all_dets.size == 0:
            all_dets = np.empty((0, 10))
        elif len(all_dets.shape) == 1:
            all_dets = np.expand_dims(all_dets, axis=0)

        seq_tp, seq_fp, seq_fn = 0, 0, 0
        seq_matched_ious = []

        # Wirtualny identyfikator klatki (zaczyna się od 1)
        virtual_frame_id = 1

        for frame_idx, img_path in frame_list:
            img_name = img_path.name
            json_path = ann_dir / f"{img_name}.json"
            
            # Pobranie GT dla klatki
            gt_boxes = load_afo_gt_mot(json_path)
            
            # Pobranie detekcji dla aktualnego virtual_frame_id i filtrowanie progowe (conf)
            frame_dets_mask = (all_dets[:, 0] == float(virtual_frame_id)) & (all_dets[:, 6] >= conf_thresh)
            frame_dets = all_dets[frame_dets_mask]
            det_boxes = frame_dets[:, 2:6].tolist()  # pobieramy tylko [tl_x, tl_y, w, h]

            # Dopasowanie boksów
            tp, fp, fn, matched_ious = match_detections_to_gt(det_boxes, gt_boxes, iou_threshold=iou_thresh)
            
            seq_tp += tp
            seq_fp += fp
            seq_fn += fn
            seq_matched_ious.extend(matched_ious)
            
            virtual_frame_id += 1

        # Obliczanie metryk dla pojedynczej sekwencji
        precision = seq_tp / (seq_tp + seq_fp) if (seq_tp + seq_fp) > 0 else 0.0
        recall = seq_tp / (seq_tp + seq_fn) if (seq_tp + seq_fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(seq_matched_ious) if seq_matched_ious else 0.0

        table_data.append([seq_id, seq_tp, seq_fp, seq_fn, f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}", f"{mean_iou:.4f}"])

        # Akumulacja globalna
        global_tp += seq_tp
        global_fp += seq_fp
        global_fn += seq_fn
        global_matched_ious.extend(seq_matched_ious)

    # Obliczanie metryk globalnych (mikro-średnia)
    glob_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    glob_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    glob_f1 = (2 * glob_precision * glob_recall) / (glob_precision + glob_recall) if (glob_precision + glob_recall) > 0 else 0.0
    glob_mean_iou = np.mean(global_matched_ious) if global_matched_ious else 0.0

    # Drukowanie tabeli sekwencji
    headers = ["Sekwencja", "TP", "FP", "FN", "Precision", "Recall", "F1-Score", "Mean IoU"]
    print("\n" + "="*85)
    print(f"RAPORT DETEKCJI Mamba-YOLO (Split: {split} | Conf Thresh: {conf_thresh} | IoU Thresh: {iou_thresh})")
    print("="*85)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Drukowanie podsumowania
    print("\n" + "="*50)
    print("PODSUMOWANIE GLOBALNE (Wszystkie Sekwencje)")
    print("="*50)
    print(f"Suma True Positives (TP):  {global_tp}")
    print(f"Suma False Positives (FP): {global_fp}")
    print(f"Suma False Negatives (FN): {global_fn}")
    print("-" * 50)
    print(f"Precyzja (Precision):      {glob_precision:.4f}")
    print(f"Pełność (Recall):          {glob_recall:.4f}")
    print(f"F1-Score:                  {glob_f1:.4f}")
    print(f"Średnie IoU (Mean IoU):    {glob_mean_iou:.4f}")
    print("="*50)


if __name__ == "__main__":
    # Ścieżki dopasuj do swojej konfiguracji
    DATA_ROOT = "data/afo"
    SPLIT = "train"
    DET_RESULTS_DIR = f"out/det_results_mamba/afo/{SPLIT}"  # lub f"out/det_results_yolox_m/afo/{SPLIT}" dla yolox_x
    
    evaluate_detector(
        data_root=DATA_ROOT,
        det_results_dir=DET_RESULTS_DIR,
        split=SPLIT,
        conf_thresh=0.5,  # Próg ufności dla boksów branych pod uwagę przy ewaluacji
        iou_thresh=0.5    # Próg akceptacji dopasowania detekcji z GT (IoU)
    )