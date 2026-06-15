import os
import cv2
import json
import glob
import torch
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.VOT.models.vgg_adnet import VGGMBackbone, ADNet
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS, ActionHistory
from models.VOT.tracking.tracking_on_sequence import rollout_one_frame
from models.VOT.utils.bbox import iou as vot_iou


def get_frame_number(filename):
    """Wyciąga numer klatki z nazwy pliku w AFO, np. z 'a_110.jpg' wyciąga 110"""
    match = re.search(r'_(\d+)\.\w+$', filename)
    return int(match.group(1)) if match else 0

def sort_key_from_filename(filepath):
    return get_frame_number(os.path.basename(filepath))

def mot_to_vot_box(mot_box):
    """Konwertuje z formatu MOT [tl_x, tl_y, w, h] na format VOT [center_x, center_y, w, h]"""
    tl_x, tl_y, w, h = mot_box
    return [tl_x + w / 2.0, tl_y + h / 2.0, w, h]

def vot_to_mot_box(vot_box):
    """Konwertuje z formatu VOT [center_x, center_y, w, h] na format MOT [tl_x, tl_y, w, h]"""
    c_x, c_y, w, h = vot_box
    return [c_x - w / 2.0, c_y - h / 2.0, w, h]

def load_afo_gt(json_path):
    """Wczytuje Ground Truth z plików JSON w formacie Supervisely (AFO)"""
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
                # Zwracamy w formacie centralnym VOT [cx, cy, w, h], żeby łatwo liczyć IoU
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                cx = min(x1, x2) + w / 2.0
                cy = min(y1, y2) + h / 2.0
                gt_boxes.append([cx, cy, w, h])
    return gt_boxes

def best_iou_match(pred_box, gt_boxes):
    """Znajduje najwyższe IoU między przewidywaną ramką a wszystkimi ramkami GT na obrazie"""
    if not gt_boxes:
        return 0.0
    ious = [vot_iou(pred_box, gt) for gt in gt_boxes]
    return max(ious)


def refine_tracking_results(
    model, 
    action_set, 
    device, 
    dataset_base_path, 
    kalman_results_dir, 
    output_dir,
    split="train",
    max_vot_steps=5,
    max_frames_per_seq=None
):
    os.makedirs(output_dir, exist_ok=True)
    seq_txt_files = sorted(glob.glob(os.path.join(kalman_results_dir, "*.txt")))
    
    total_kalman_ious = []
    total_adnet_ious = []
    
    for txt_file in seq_txt_files:
        seq_name = os.path.basename(txt_file).replace('.txt', '')
        logger.info(f"\nRozpoczynam korektę dla sekwencji: {seq_name}")
        
        # Przygotowanie ścieżek dla AFO
        img_dir = os.path.join(dataset_base_path, split, "img")
        ann_dir = os.path.join(dataset_base_path, split, "ann")
        
        # Wczytanie obrazów i sortowanie po numerze klatki (np. a_110.jpg)
        seq_images = sorted(glob.glob(os.path.join(img_dir, f"{seq_name}_*.*")), key=sort_key_from_filename)
        
        # Wczytanie przewidywań z pliku .txt (MOT format)
        preds = np.loadtxt(txt_file, delimiter=',')
        if preds.size == 0:
            continue
            
        max_frame_id = int(preds[:, 0].max())
        if max_frames_per_seq:
            max_frame_id = min(max_frame_id, max_frames_per_seq)
            
        track_histories = {} # Słownik: track_id -> ActionHistory
        refined_results = []
        
        seq_kalman_ious = []
        seq_adnet_ious = []

        process_bar = tqdm(range(1, max_frame_id + 1), desc="Klatki")
        
        for frame_id in process_bar:
            # Szukamy odpowiadającego obrazka. W trackerze użyto indeksu (frame_id - 1)
            if frame_id > len(seq_images):
                continue
                
            img_path = seq_images[frame_id - 1]
            img_name = os.path.basename(img_path)
            json_path = os.path.join(ann_dir, f"{img_name}.json")
            
            # Odczyt GT dla tej klatki (do ewaluacji)
            gt_boxes_vot = load_afo_gt(json_path)
            
            frame_img = cv2.imread(img_path)
            if frame_img is None:
                continue
                
            # Wyciągamy detekcje Kalmana tylko dla obecnej klatki
            current_dets = preds[preds[:, 0] == float(frame_id)]
            
            for det in current_dets:
                _, track_id, tl_x, tl_y, w, h, score, _, _, _ = det
                track_id = int(track_id)
                
                # Format: od MOT do VOT
                mot_box = [tl_x, tl_y, w, h]
                initial_box_vot = mot_to_vot_box(mot_box)
                
                # Inicjalizacja historii dla nowego obiektu
                if track_id not in track_histories:
                    track_histories[track_id] = ActionHistory(action_set, history_length=10)
                
                # KOREKTA PRZEZ ADNET
                result = rollout_one_frame(
                    model=model,
                    frame=frame_img,
                    initial_box=initial_box_vot,
                    gt_box=initial_box_vot, # Dummy GT, bo rollout go potrzebuje do rewardu, tu nas nie obchodzi
                    action_set=action_set,
                    device=device,
                    history=track_histories[track_id],
                    max_steps=max_vot_steps, 
                    sample_actions=False
                )
                
                final_box_vot = result["final_box"]
                track_histories[track_id] = result["history"] # Zapisanie historii na kolejną klatkę
                
                # Ewaluacja (Obliczanie metryk w locie)
                if gt_boxes_vot:
                    iou_kalman = best_iou_match(initial_box_vot, gt_boxes_vot)
                    iou_adnet = best_iou_match(final_box_vot, gt_boxes_vot)
                    
                    seq_kalman_ious.append(iou_kalman)
                    seq_adnet_ious.append(iou_adnet)
                
                # Konwersja z powrotem do MOT format i zapis
                ref_tl_x, ref_tl_y, ref_w, ref_h = vot_to_mot_box(final_box_vot)
                refined_results.append(f"{frame_id},{track_id},{ref_tl_x:.2f},{ref_tl_y:.2f},{ref_w:.2f},{ref_h:.2f},{score:.2f},-1,-1,-1\n")
        
        # Zapis do pliku
        out_txt_path = os.path.join(output_dir, f"{seq_name}.txt")
        with open(out_txt_path, 'w') as f:
            f.writelines(refined_results)
            
        # Logowanie wyników dla sekwencji
        if seq_kalman_ious:
            mean_k = np.mean(seq_kalman_ious)
            mean_a = np.mean(seq_adnet_ious)
            diff = (mean_a - mean_k) * 100
            
            logger.info(f"--- Wyniki dla {seq_name} ---")
            logger.info(f"Średnie IoU (Kalman): {mean_k:.4f}")
            logger.info(f"Średnie IoU (ADNet):  {mean_a:.4f} | Różnica: {diff:+.2f}%")
            
            total_kalman_ious.extend(seq_kalman_ious)
            total_adnet_ious.extend(seq_adnet_ious)

    # Logowanie wyników globalnych
    if total_kalman_ious:
        tot_mean_k = np.mean(total_kalman_ious)
        tot_mean_a = np.mean(total_adnet_ious)
        tot_succ_k = np.mean([v > 0.5 for v in total_kalman_ious])
        tot_succ_a = np.mean([v > 0.5 for v in total_adnet_ious])
        
        print("\n" + "="*50)
        print("RAPORT KOŃCOWY: KALMAN vs ADNET (Post-Processing)")
        print("="*50)
        print(f"Ilość przeanalizowanych ramek: {len(total_kalman_ious)}")
        print(f"Średnie IoU przed (Kalman): {tot_mean_k:.4f}")
        print(f"Średnie IoU po (ADNet)  : {tot_mean_a:.4f}")
        print(f"Poprawa IoU: {((tot_mean_a / tot_mean_k) - 1) * 100:+.2f}%\n")
        
        print(f"Skuteczność (IoU > 0.5) przed: {tot_succ_k:.4f}")
        print(f"Skuteczność (IoU > 0.5) po   : {tot_succ_a:.4f}")
        print(f"Poprawa skuteczności: {((tot_succ_a / tot_succ_k) - 1) * 100:+.2f}%")
        print("="*50)

if __name__ == "__main__":
    # 1. Konfiguracja urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_set = ORIGINAL_ADNET_ACTIONS
    
    # 2. Inicjalizacja modelu
    backbone = VGGMBackbone()
    model = ADNet(num_actions=len(action_set), history_length=10, backbone=backbone).to(device)
    
    # Wczytanie wag (zmień ścieżkę na właściwą dla Twoich najlepszych wag)
    checkpoint_location = "checkpoints/adnet_sl_vgg_best_1.pt"
    checkpoint = torch.load(checkpoint_location, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Parametry ścieżek zbioru danych
    DATASET_BASE = "data/afo/"
    KALMAN_RESULTS = "out/track_res/afo/train/afo_byte/"
    OUTPUT_RESULTS = "out/track_res/afo/train/afo_byte_refined/"
    
    # Uruchomienie korekty
    refine_tracking_results(
        model=model,
        action_set=action_set,
        device=device,
        dataset_base_path=DATASET_BASE,
        kalman_results_dir=KALMAN_RESULTS,
        output_dir=OUTPUT_RESULTS,
        max_vot_steps=10,
        split="train"
    )
