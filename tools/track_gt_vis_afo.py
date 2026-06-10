"""
Wizualizacja adnotacji Ground Truth (GT) z danymi o śledzeniu obiektów.
"""

import argparse
import os
import json
import cv2
from loguru import logger
from tqdm import tqdm
import supervisely as sly

def get_args():
    parser = argparse.ArgumentParser(description="Wizualizacja GT dla zbioru AFO")
    parser.add_argument('--data_root', type=str, default='data/afo', help='Katalog z projektem Supervisely')
    parser.add_argument('--save_dir', type=str, default='out/gt_vis', help='Folder wyjściowy na zdjęcia z ramkami')
    parser.add_argument('--split', type=str, default=None, help='Opcjonalnie: konkretny podział do wizualizacji (np. train, valid)')
    return parser.parse_args()

def get_color(idx):
    """
    Identyczna funkcja do generowania kolorów jak w algorytmach trackujących.
    Gwarantuje, że to samo ID zawsze otrzyma ten sam kolor.
    """
    idx = int(idx) * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def main(args):
    # Wczytanie projektu za pomocą Supervisely
    project = sly.Project(args.data_root, sly.OpenMode.READ)
    logger.info(f"Otwarto projekt: {project.name} (Łącznie obrazów: {project.total_items})")

    # Przechodzimy przez wszystkie datasety (czyli np. train, valid, test)
    for dataset in project.datasets:
        # Jeśli użytkownik podał konkretny split w argumentach, pomijamy resztę
        if args.split and dataset.name != args.split:
            continue
            
        logger.info(f"\nRozpoczynam renderowanie zestawu: {dataset.name}")
        
        # Pobieramy elementy i sortujemy alfabetycznie wg nazw plików (np. a_001.jpg)
        items = list(dataset.items())
        items.sort(key=lambda x: x[0])

        process_bar = tqdm(items, desc=f"Wizualizacja {dataset.name}", ncols=120)

        for item_name, image_path, ann_path in process_bar:
            
            # Ekstrakcja nazwy sekwencji na potrzeby tworzenia ładnych podfolderów (np. 'a' z 'a_001.jpg')
            stem = os.path.splitext(item_name)[0]
            parts = stem.split("_")
            seq_id = "_".join(parts[:-1]) if len(parts) >= 2 else "unknown_seq"
                
            seq_save_dir = os.path.join(args.save_dir, dataset.name, seq_id)
            if not os.path.exists(seq_save_dir):
                os.makedirs(seq_save_dir)
            
            # Wczytanie obrazka za pomocą CV2 (zamiast sly.image.read, by uniknąć problemów BGR vs RGB)
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Nie można wczytać obrazu: {image_path}")
                continue
                
            # Ręczne wczytanie i parsowanie pliku JSON z adnotacją
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_json = json.load(f)
                
            # Rysowanie obiektów z tego konkretnego pliku
            for obj in ann_json.get("objects", []):
                if obj.get("geometryType") != "rectangle":
                    continue
                    
                # Pobranie docelowego identyfikatora obiektu
                obj_id = obj.get("id")
                
                # Zabezpieczenie przed brakiem współrzędnych
                pts = obj.get("points", {}).get("exterior", [])
                if len(pts) < 2:
                    continue
                    
                # Ekstrakcja X i Y ze zagnieżdżonej listy
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                
                # Format Top-Left i Bottom-Right 
                pt1 = (min(x_coords), min(y_coords))
                pt2 = (max(x_coords), max(y_coords))
                
                # Wygenerowanie stałego koloru na podstawie numeru ID
                color = get_color(obj_id)
                
                # 1. Rysowanie samej ramki (bounding box)
                cv2.rectangle(img, pt1, pt2, color, thickness=3)
                
                # 2. Przygotowanie tekstu (tylko ID obiektu)
                text = f'id:{obj_id}'
                
                # 3. Dodanie czytelnego tła za tekstem i naniesienie go
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)
                cv2.rectangle(img, (pt1[0], pt1[1] - text_h - 6), (pt1[0] + text_w, pt1[1]), color, -1)
                cv2.putText(img, text, (pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                
            # Zapis wyrenderowanego obrazu z ramkami do folderu wyjściowego
            save_path = os.path.join(seq_save_dir, item_name)
            cv2.imwrite(save_path, img)

    logger.info(f"Zakończono! Wszystkie wyrenderowane obrazy znajdziesz w: {args.save_dir}")

if __name__ == '__main__':
    args = get_args()
    main(args)