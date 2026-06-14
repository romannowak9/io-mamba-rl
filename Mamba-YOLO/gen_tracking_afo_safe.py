import os
import re
import sys
import torch
import numpy as np
from glob import glob
from pathlib import Path
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO

# Zabezpieczenie bezpiecznego ładowania obiektów (PyTorch 2.6+)
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

def extract_main_group(filename):
    """Wyciąga główny identyfikator (np. 'a' z 'a_102_001.jpg')."""
    basename = os.path.basename(filename)
    match = re.match(r'^([a-zA-Z0-9]+)_', basename)
    if match:
        return match.group(1)
    return "unknown"

def main():
    # --- JAWNIE ZDEFINIOWANE ARGUMENTY WEJŚCIOWE ---
    model_path = "/home/luki10101/projects/Mamba-YOLO/output_dir/afo_project/mambayolo_afo_final_run/weights/best.pt"
    images_dir = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/train/images" 
    output_dir = "./output_dir/tracking_txt_results"
    
    img_size = 1280  
    conf_thresh = 0.05  # Niski próg zgodny z Twoim wzorcowym skryptem
    
    os.makedirs(output_dir, exist_ok=True)

    logger.info("⏳ Ładowanie modelu Mamba-YOLO...")
    model = YOLO(model_path)
    logger.info("✅ Wagi modelu załadowane pomyślnie.")

    logger.info(f"📂 Skanowanie katalogu z obrazami: {images_dir}")
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.PNG')
    all_images = []
    for ext in extensions:
        all_images.extend(glob(os.path.join(images_dir, ext)))
        
    if not all_images:
        logger.error(f"❌ Nie znaleziono żadnych zdjęć w lokalizacji: {images_dir}")
        return

    # Grupowanie (a_102, a_103 wpadają pod wspólny klucz 'a')
    groups = defaultdict(list)
    for img_path in all_images:
        group_name = extract_main_group(img_path)
        groups[group_name].append(img_path)

    logger.info(f"📊 Wykryto {len(groups)} głównych grup do przetworzenia.")

    # URUCHOMIENIE W TRYBIE BEZ GRADIENTÓW (Zabezpieczenie pamięci GPU)
    with torch.no_grad():
        for group_idx, (group_name, group_frames) in enumerate(groups.items(), 1):
            
            # KRYTYCZNE: Sortowanie gwarantuje, że klatki idą w kolejności chronologicznej
            group_frames.sort()
            
            out_txt_path = os.path.join(output_dir, f"{group_name}.txt")
            logger.info(f"🎬 [{group_idx}/{len(groups)}] Przetwarzanie grupy: '{group_name}' ({len(group_frames)} klatek)")
            
            with open(out_txt_path, 'w') as f_out:
                # Iterujemy klatka po klatce przez listę ścieżek
                for frame_idx, img_path in enumerate(tqdm(group_frames, desc=f"Grupa {group_name}")):
                    virtual_frame_id = frame_idx + 1
                    
                    # WYWOŁANIE MODELU - dokładnie takie parametry, jakie zadziałały w teście diagnostycznym:
                    # 1. source=img_path (podajemy jako pojedynczy string, nie listę, co zapobiega przepełnieniu)
                    # 2. half=False (pełna precyzja FP32 - eliminuje crash sterownika CUDA w WSL)
                    # 3. persist=True (bardzo ważne: mówi trackerowi, że to kontynuacja poprzedniego pliku)
                    results = model.track(
                        source=img_path,
                        imgsz=img_size,
                        conf=conf_thresh,
                        device=0,
                        half=False,       # Ustawienie FP32 (dokładnie tak jak w udanym teście)
                        tracker="bytetrack.yaml",
                        persist=True,     # Utrzymuje ciągłość Filtru Kalmana dla tej grupy
                        verbose=False
                    )
                    
                    result = results[0]
                    
                    if result.boxes is not None:
                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  
                        clss = result.boxes.cls.int().cpu().numpy()   
                        confs = result.boxes.conf.cpu().numpy()       

                        for box, cls_id, conf in zip(boxes_xyxy, clss, confs):
                            x1, y1, x2, y2 = box
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Format wyjściowy MOT bez ID obiektu:
                            # [numer_klatki],[klasa],[x_min],[y_min],[szerokość],[wysokość],[pewność]
                            line = f"{virtual_frame_id},{cls_id},{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{conf:.4f}\n"
                            f_out.write(line)
                    
                    # Ręczne czyszczenie pamięci podręcznej co 20 klatek, by dać odetchnąć karcie w WSL
                    if virtual_frame_id % 20 == 0:
                        torch.cuda.empty_cache()

            logger.info(f"✅ Zapisano wyniki do pliku: {out_txt_path}")
            
            # Pełne czyszczenie cache po zakończeniu całej grupy przed przejściem do kolejnej
            torch.cuda.empty_cache()

    logger.info(f"🎉 Wszystkie dane zostały pomyślnie zapisane w katalogu: {output_dir}")

if __name__ == '__main__':
    main()