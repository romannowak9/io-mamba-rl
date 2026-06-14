#!/usr/bin/env python3
"""
Track from saved det results using Ultralytics BYTETracker engine.
Outputs tracking visualizations in native high resolution.
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
from loguru import logger
from tqdm import tqdm
import re
from ultralytics import YOLO

# Bezpieczne parsowanie nazw plików z Ultralytics (fallback jeśli brak helperów)
def sort_key_from_filename(filename):
    basename = os.path.basename(filename)
    numbers = [int(s) for s in re.findall(r'\d+', basename)]
    return numbers[-1] if numbers else basename

# Import wbudowanego silnika Ultralytics
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


def get_args():
    parser = argparse.ArgumentParser(description="Mamba-YOLO Offline ByteTrack Potok (Rozdzielczość Natywna)")
    
    # Główne parametry ścieżek
    parser.add_argument('--det_path', required=True, type=str, 
                        help="Ścieżka do katalogu z plikami detekcji .txt z Mamba-YOLO")
    parser.add_argument('--motion', type=str, default='byte', help="Nazwa algorytmu asocjacji")
    
    # Parametry konfiguracyjne ByteTracka
    parser.add_argument('--track_high_thresh', type=float, default=0.5, help="Próg dla silnych detekcji")
    parser.add_argument('--track_low_thresh', type=float, default=0.1, help="Próg dla słabych detekcji")
    parser.add_argument('--new_track_thresh', type=float, default=0.6, help="Próg inicjalizacji nowej trajektorii")
    parser.add_argument('--track_buffer', type=int, default=30, help="Pamięć trackera (liczba klatek)")
    parser.add_argument('--match_thresh', type=float, default=0.8, help="Próg dopasowania IoU")
    
    # Parametry wizualizacji i zapisu
    parser.add_argument('--data_root', type=str, 
                        default="/home/luki10101/projects/Mamba-YOLO/datasets/afo/validation/images",
                        help="Katalog z oryginalnymi obrazami bazy danych")
    parser.add_argument('--vis', action='store_true', help="Czy generować klatki wideo z narysowanymi boksami i ID")
    parser.add_argument('--save_dir', type=str, default='track_results/{dataset_name}/{split}',
                        help="Struktura katalogu wyjściowego")

    return parser.parse_args()


def save_results(save_dir, folder_name, seq_name, results):
    """Zapisuje końcowy wynik śledzenia do pliku tekstowego w standardzie MOT."""
    out_folder = os.path.join(save_dir, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    out_file_path = os.path.join(out_folder, f"{seq_name}.txt")
    
    with open(out_file_path, 'w') as f:
        for frame_id, target_ids, tlwhs, clses, scores in results:
            for tid, tlwh, score, cls_id in zip(target_ids, tlwhs, scores, clses):
                f.write(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},{cls_id},-1,-1\n")
                
    return out_folder


def get_color(idx):
    """Generuje unikalny i stały kolor BGR dla danego ID obiektu."""
    idx = int(idx) * 7  # Zmiana mnożnika dla lepszej dywersyfikacji kolorów bliskich ID
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_img(img, frame_id, results, save_dir):
    """Rysuje ramki otaczające, ID oraz klasy bezpośrednio w natywnej rozdzielczości kadru."""
    os.makedirs(save_dir, exist_ok=True)
    assert img is not None

    img_ = np.ascontiguousarray(np.copy(img))
    tlwhs, ids, clses, scores = results[0], results[1], results[2], results[3]
    
    # Dynamiczne dopasowanie grubości linii i czcionki do wielkich rozdzielczości (np. 4K)
    h, w = img_.shape[:2]
    thickness = max(2, int(w / 1000))
    font_scale = max(0.5, w / 2000)

    for tlwh, target_id, cls_id, score in zip(tlwhs, ids, clses, scores):
        tlbr = (int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
        color = get_color(target_id)
        
        # Rysowanie ramki obiektu
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], color, thickness=thickness)
        
        # Przygotowanie podpisu: ID, Klasa i Pewność
        text = f"ID:{target_id} Cls:{int(cls_id)} ({score:.2f})"
        cv2.putText(img_, text, (tlbr[0], max(20, tlbr[1] - 8)), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, 
                    color=(255, 255, 255), thickness=max(1, int(thickness/2)), lineType=cv2.LINE_AA)
        
    cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.jpg"), img_)


def main(args):
    exp_infos = os.path.normpath(args.det_path).split(os.sep)
    dataset_name = exp_infos[-2] if len(exp_infos) > 1 else "dataset"
    split = exp_infos[-1] if len(exp_infos) > 0 else "split"
    
    seq_dets = sorted(os.listdir(args.det_path))
    save_dir = args.save_dir.format(dataset_name=dataset_name, split=split)

    for seq in seq_dets:
        if not seq.endswith('.txt') or 'meta_data' in seq:
            continue

        seq_name = seq[:-4]
        logger.info(f"🚀 Śledzenie sekwencji: {seq}")
        
        img_base_dir = args.data_root
        seq_images = sorted(glob.glob(os.path.join(img_base_dir, f"{seq_name}_*.*")), key=sort_key_from_filename)

        file_name = os.path.join(args.det_path, seq)
        
        if os.path.getsize(file_name) == 0:
            logger.warning(f"⚠️ Pomijam pusty plik detekcji: {file_name}")
            continue
            
        file_content = np.loadtxt(file_name, dtype=float, delimiter=',')
        if file_content.ndim == 1:
            file_content = np.expand_dims(file_content, axis=0)

        max_frame_id = int(file_content[:, 0].max())

        tracker_cfg = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=args.track_high_thresh,
            track_low_thresh=args.track_low_thresh,
            new_track_thresh=args.new_track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            frame_rate=30
        )

        model_path = "/home/luki10101/projects/Mamba-YOLO/output_dir/afo_project/mambayolo_afo_final_run/weights/best.pt"
        
        tracker = YOLO(model_path).tracker(tracker_cfg)
        results = []

        for frame_id in tqdm(range(1, max_frame_id + 1), desc=f"Sekwencja {seq_name}", ncols=120):
            current_det = file_content[file_content[:, 0] == float(frame_id)]
            cur_tlwh, cur_id, cur_cls, cur_score = [], [], [], []
            
            if len(current_det) > 0:
                cls_ids = current_det[:, 1]
                x1 = current_det[:, 2]
                y1 = current_det[:, 3]
                w = current_det[:, 4]
                h = current_det[:, 5]
                confs = current_det[:, 6]
                
                x_center = x1 + (w / 2.0)
                y_center = y1 + (h / 2.0)
                
                stack_xyxy = np.stack([x1, y1, x1 + w, y1 + h], axis=1)
                stack_xywh = np.stack([x_center, y_center, w, h], axis=1)
                
                # Ochrona przed awarią jednowymiarowych tensorów (Gwarancja stabilności dla N=1)
                if len(current_det) == 1:
                    dummy_xyxy = np.array([[-1000.0, -1000.0, -1000.0, -1000.0]])
                    dummy_xywh = np.array([[-1000.0, -1000.0, 0.0, 0.0]])
                    dummy_conf = np.array([0.0])
                    dummy_cls = np.array([cls_ids[0]])
                    
                    stack_xyxy = np.vstack([stack_xyxy, dummy_xyxy])
                    stack_xywh = np.vstack([stack_xywh, dummy_xywh])
                    confs = np.concatenate([confs, dummy_conf])
                    cls_ids = np.concatenate([cls_ids, dummy_cls])
                
                bboxes_xyxy = torch.from_numpy(stack_xyxy).float()
                bboxes_xywh = torch.from_numpy(stack_xywh).float()
                confs_tensor = torch.from_numpy(confs).float()
                cls_ids_tensor = torch.from_numpy(cls_ids).float()
                
                dummy_result = IterableSimpleNamespace(
                    boxes=bboxes_xyxy,
                    xyxy=bboxes_xyxy,
                    xywh=bboxes_xywh,
                    conf=confs_tensor,
                    cls=cls_ids_tensor
                )
                
                output_tracklets = tracker.update(dummy_result)
                
                if output_tracklets is not None and len(output_tracklets) > 0:
                    for obj in output_tracklets:
                        if len(obj) >= 5:
                            rx1, ry1, rx2, ry2 = obj[0], obj[1], obj[2], obj[3]
                            tid = int(obj[4])
                            score = obj[5] if len(obj) > 5 else 1.0
                            cid = int(obj[6]) if len(obj) > 6 else 0
                            
                            rw = rx2 - rx1
                            rh = ry2 - ry1
                            
                            cur_tlwh.append([rx1, ry1, rw, rh])
                            cur_id.append(tid)
                            cur_cls.append(cid)
                            cur_score.append(score)

            results.append((frame_id, cur_id, cur_tlwh, cur_cls, cur_score))

            # Sekcja wizualizacji: Rysowanie bezpośrednio na oryginalnym obrazie bez resizingu
            if args.vis and len(cur_id) > 0:
                if frame_id <= len(seq_images):
                    cur_frame_path = seq_images[frame_id - 1]
                    cur_frame = cv2.imread(cur_frame_path)
                    
                    if cur_frame is not None:
                        plot_img(img=cur_frame, frame_id=frame_id, 
                                 results=[cur_tlwh, cur_id, cur_cls, cur_score], 
                                 save_dir=os.path.join(save_dir, 'vis_results', seq_name))

        # Zapis głównego pliku tekstowego MOT
        folder_name = f"{dataset_name}_{args.motion}"
        save_results(save_dir=save_dir, folder_name=folder_name, 
                     seq_name=seq_name, results=results)
        logger.info(f"✅ Zapisano plik tekstowy: {os.path.join(save_dir, folder_name, seq_name + '.txt')}")

        # Automatyczna kompilacja klatek w płynne wideo MP4 (OpenCV VideoWriter)
        if args.vis:
            vis_seq_dir = os.path.join(save_dir, 'vis_results', seq_name)
            output_video_dir = os.path.join(save_dir, 'tracking_videos')
            os.makedirs(output_video_dir, exist_ok=True)
            output_mp4_path = os.path.join(output_video_dir, f"{seq_name}_tracked.mp4")
            
            images_to_video = sorted(glob.glob(os.path.join(vis_seq_dir, "*.jpg")))
            if images_to_video:
                sample_frame = cv2.imread(images_to_video[0])
                height, width, _ = sample_frame.shape
                
                # Używamy uniwersalnego kodeka mp4v dla czystych plików MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_mp4_path, fourcc, 30, (width, height))
                
                logger.info(f"🎬 Kompilowanie wideo MP4 w rozdzielczości {width}x{height}...")
                for img_p in images_to_video:
                    video_writer.write(cv2.imread(img_p))
                video_writer.release()
                logger.info(f"🍿 Sukces! Gotowy film zapisano w: {output_mp4_path}")


if __name__ == '__main__':
    args = get_args()
    main(args)