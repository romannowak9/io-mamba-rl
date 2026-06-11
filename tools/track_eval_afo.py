"""
Skrypt do ewaluacji wyników śledzenia (Tracking) na zbiorze AFO przy użyciu motmetrics.
Raportuje zaawansowane metryki śledzenia oraz metrykę Det-MOTA (czysta detekcja bez kar za ID Switches).
"""

import argparse
import os
import glob
import json
import numpy as np
import motmetrics as mm
from loguru import logger

def get_args():
    parser = argparse.ArgumentParser(description="Ewaluacja śledzenia AFO za pomocą motmetrics")
    parser.add_argument('--track_path', required=True, type=str, help='Ścieżka do plików z wynikami śledzenia (np. track_results/afo_byte/train)')
    parser.add_argument('--data_root', required=True, type=str, help='Główny folder z podziałem AFO (zawiera folder ann/)')
    parser.add_argument('--conf_thresh', type=float, default=0.0, help='Minimalny próg ufności (jeśli chcemy dodatkowo filtrować tracki)')
    return parser.parse_args()

def load_track_txt(path, conf_thresh=0.0):
    if not os.path.exists(path):
        return {}
    try:
        data = np.loadtxt(path, delimiter=',', ndmin=2)
        if len(data) == 0:
            return {}
        if conf_thresh > 0:
            data = data[data[:, 6] >= conf_thresh]
        return {int(f_id): data[data[:, 0] == f_id] for f_id in np.unique(data[:, 0])}
    except Exception as e:
        logger.error(f"Błąd ładowania wyników śledzenia {path}: {e}")
        return {}

def parse_supervisely_gt_to_mot(ann_dir, seq_id):
    ann_files = glob.glob(os.path.join(ann_dir, f"{seq_id}_*.json"))
    
    def get_frame_num(path):
        stem = os.path.basename(path).replace(".jpg.json", "").replace(".json", "")
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return 0
            
    ann_files.sort(key=get_frame_num)
    
    gt_dict = {}
    for virtual_frame_id, ann_path in enumerate(ann_files, start=1):
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann_json = json.load(f)
            
        frame_boxes = []
        for obj in ann_json.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue
                
            pts = obj.get("points", {}).get("exterior", [])
            if len(pts) < 2:
                continue
                
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            w = float(x_max - x_min)
            h = float(y_max - y_min)
            
            obj_id = obj.get("id", -1)
            frame_boxes.append([obj_id, x_min, y_min, w, h])
            
        if frame_boxes:
            gt_dict[virtual_frame_id] = np.array(frame_boxes)
            
    return gt_dict

def main(args):
    ann_dir = os.path.join(args.data_root, "ann")
    if not os.path.exists(ann_dir):
        logger.error(f"Katalog adnotacji nie istnieje: {ann_dir}")
        return

    accs = []
    names = []
    
    track_files = sorted(glob.glob(os.path.join(args.track_path, "*.txt")))
    if not track_files:
        logger.error(f"Nie znaleziono plików .txt w: {args.track_path}")
        return

    logger.info(f"Rozpoczynam ewaluację śledzenia dla {len(track_files)} sekwencji...")

    for track_file in track_files:
        seq_id = os.path.splitext(os.path.basename(track_file))[0]
        
        preds = load_track_txt(track_file, args.conf_thresh)
        gts = parse_supervisely_gt_to_mot(ann_dir, seq_id)
        
        if not gts:
            logger.warning(f"Brak adnotacji GT dla sekwencji: {seq_id}. Pomijam.")
            continue
            
        acc = mm.MOTAccumulator(auto_id=True)
        max_frame = max(max(preds.keys(), default=0), max(gts.keys(), default=0))
        
        for frame_id in range(1, max_frame + 1):
            gt_frame = gts.get(frame_id, np.empty((0, 5)))
            pred_frame = preds.get(frame_id, np.empty((0, 10)))
            
            gt_ids = gt_frame[:, 0].astype(int) if len(gt_frame) > 0 else []
            gt_boxes = gt_frame[:, 1:5] if len(gt_frame) > 0 else np.empty((0, 4))
            
            pred_ids = pred_frame[:, 1].astype(int) if len(pred_frame) > 0 else []
            pred_boxes = pred_frame[:, 2:6] if len(pred_frame) > 0 else np.empty((0, 4))
            
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, distances)
            
        accs.append(acc)
        names.append(seq_id)

    if not accs:
        logger.error("Brak danych do wyliczenia metryk.")
        return

    # Wyliczamy bazowe metryki z motmetrics, w tym te niezbędne do Det-MOTA (FP, FN, Obj)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, 
        names=names, 
        metrics=['idf1', 'mota', 'num_switches', 'mostly_tracked', 'mostly_lost', 'precision', 'recall', 'num_false_positives', 'num_misses', 'num_objects'],
        generate_overall=True
    )
    
    # --- OBLICZANIE DET-MOTA ---
    # Obliczamy na podstawie: 1.0 - (False_Positives + False_Negatives) / Total_Objects
    summary['det_mota'] = summary.apply(
        lambda row: 1.0 - (row['num_false_positives'] + row['num_misses']) / row['num_objects'] 
        if row['num_objects'] > 0 else 0.0, 
        axis=1
    )
    
    # Przypisujemy formatowanie % do nowej kolumny
    mh.formatters['det_mota'] = mm.io.Formatters['mota']

    # Ograniczamy widok tabeli tylko do kluczowych kolumn, żeby zachować czytelność
    cols_to_show = ['idf1', 'mota', 'det_mota', 'num_switches', 'mostly_tracked', 'mostly_lost', 'precision', 'recall']
    summary_to_show = summary[cols_to_show]
    
    # Renderowanie raportu
    str_summary = mm.io.render_summary(
        summary_to_show, 
        formatters=mh.formatters, 
        namemap={
            'idf1': 'IDF1', 
            'mota': 'MOTA', 
            'det_mota': 'Det-MOTA', 
            'num_switches': 'ID Sw.', 
            'mostly_tracked': 'MT', 
            'mostly_lost': 'ML',
            'precision': 'Prec',
            'recall': 'Rec'
        }
    )
    
    print("\n--- RAPORT EWALUACJI ŚLEDZENIA ORAZ DETEKCJI ---")
    print(str_summary)

if __name__ == '__main__':
    args = get_args()
    main(args)