"""
Skrypt do ewaluacji detekcji zbioru AFO w formacie MOTChallenge przy użyciu motmetrics.
"""

import argparse
import os
import glob
import json
import numpy as np
import motmetrics as mm
from loguru import logger
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Ewaluacja detekcji AFO za pomocą motmetrics")
    parser.add_argument('--det_path', required=True, type=str, help='Ścieżka do wygenerowanych plików .txt (np. out/det_results/afo/train)')
    parser.add_argument('--data_root', required=True, type=str, help='Główny folder z podziałem AFO (zawiera folder ann/)')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Próg pewności (Confidence), powyżej którego detekcje są brane pod uwagę')
    return parser.parse_args()

def load_mot_txt(path, conf_thresh=0.0):
    """Wczytuje plik tekstowy MOT i filtruje po progu ufności."""
    if not os.path.exists(path):
        return {}
    try:
        data = np.loadtxt(path, delimiter=',', ndmin=2)
        if len(data) == 0:
            return {}
        # Filtrowanie po confidence (kolumna indeks 6)
        data = data[data[:, 6] >= conf_thresh]
        
        # Grupowanie według frame_id
        return {int(f_id): data[data[:, 0] == f_id] for f_id in np.unique(data[:, 0])}
    except Exception as e:
        logger.error(f"Błąd ładowania predykcji {path}: {e}")
        return {}

def parse_supervisely_gt_to_mot(ann_dir, seq_id):
    """Konwertuje pliki JSON danej sekwencji na strukturę MOT w pamięci."""
    # Szukamy plików adnotacji dla wybranej sekwencji
    ann_files = glob.glob(os.path.join(ann_dir, f"{seq_id}_*.json"))
    
    def get_frame_num(path):
        # Pobieranie numeru klatki z nazwy pliku np. seq_a_001.jpg.json -> 1
        stem = os.path.basename(path).replace(".jpg.json", "").replace(".json", "")
        try:
            return int(stem.split('_')[-1])
        except ValueError:
            return 0
            
    ann_files.sort(key=get_frame_num)
    
    gt_dict = {}
    # Podobnie jak w gen_det_afo, mapujemy klatki na virtual_frame_id (1, 2, 3...)
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
            
            # Unikalne ID obiektu na klatce (w formacie obrazowym traktowane jako dummy ID)
            obj_id = obj.get("id", 1) 
            
            frame_boxes.append([obj_id, x_min, y_min, w, h])
            
        if frame_boxes:
            gt_dict[virtual_frame_id] = np.array(frame_boxes)
            
    return gt_dict

def main(args):
    ann_dir = os.path.join(args.data_root, "ann")
    if not os.path.exists(ann_dir):
        logger.error(f"Katalog adnotacji nie istnieje: {ann_dir}")
        return

    # Inicjalizacja akumulatora motmetrics
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Pobieramy pliki predykcji tekstowych
    det_files = sorted(glob.glob(os.path.join(args.det_path, "*.txt")))
    det_files = [f for f in det_files if "meta_data" not in f]

    if not det_files:
        logger.error(f"Nie znaleziono plików tekstowych z detekcjami w: {args.det_path}")
        return

    logger.info(f"Rozpoczynam ewaluację dla {len(det_files)} sekwencji...")

    for det_file in det_files:
        seq_id = os.path.splitext(os.path.basename(det_file))[0]
        
        # 1. Ładowanie predykcji i GT
        preds = load_mot_txt(det_file, args.conf_thresh)
        gts = parse_supervisely_gt_to_mot(ann_dir, seq_id)
        
        if not gts:
            logger.warning(f"Brak adnotacji GT dla sekwencji: {seq_id}. Pomijam.")
            continue
            
        # Wyznaczamy wspólny zakres klatek
        max_frame = max(max(preds.keys(), default=0), max(gts.keys(), default=0))
        
        # 2. Rejestrowanie zdarzeń klatka po klatce w motmetrics
        for frame_id in range(1, max_frame + 1):
            gt_frame = gts.get(frame_id, np.empty((0, 5)))
            pred_frame = preds.get(frame_id, np.empty((0, 10)))
            
            # Wyciągamy identyfikatory i współrzędne (format: x, y, w, h)
            gt_ids = gt_frame[:, 0].astype(int) if len(gt_frame) > 0 else []
            gt_boxes = gt_frame[:, 1:5] if len(gt_frame) > 0 else np.empty((0, 4))
            
            # Dla predykcji generujemy dummy_id (bo to surowy detektor bez trackera)
            pred_ids = np.arange(len(pred_frame))
            pred_boxes = pred_frame[:, 2:6] if len(pred_frame) > 0 else np.empty((0, 4))
            
            # Obliczanie macierzy odległości IoU (Intersection over Union) między GT a predykcją
            # Maksymalny dopuszczalny dystans IoU = 0.5 (czyli nakładanie min. 50%)
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            
            # Aktualizacja stanu akumulatora dla danej klatki w sekwencji
            acc.update(gt_ids, pred_ids, distances)

    # 3. Generowanie raportu końcowego
    logger.info("Generowanie raportu metryk...")
    mh = mm.metrics.create()
    
    summary = mh.compute(
        acc, 
        metrics=['num_frames', 'mota', 'precision', 'recall', 'idp', 'idr'], 
        name='AFO_YOLOX_Evaluation'
    )
    
    # Sformatowanie wyników do ładnej tabeli
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={'num_frames': 'Frames', 'mota': 'MOTA', 'precision': 'Precision', 'recall': 'Recall'}
    )
    print("\n" + str_summary)

if __name__ == '__main__':
    args = get_args()
    main(args)