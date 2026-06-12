"""
Skrypt do generowania wizualizacji śledzenia (nakładanie ramek) 
dostosowany do płaskiej struktury zbioru AFO (np. klatki o nazwach 'd_001.jpg').
"""

import os
import glob
import argparse
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Wizualizacja gotowych wyników śledzenia z plików TXT")
    parser.add_argument('--track_dir', required=True, type=str, 
                        help='Katalog z plikami .txt (np. track_res/afo/train/afo_byte/)')
    parser.add_argument('--img_root', required=True, type=str, 
                        help='Katalog ze zdjęciami (np. data/afo/train/ lub data/afo/train/img/)')
    parser.add_argument('--save_dir', type=str, default='vis_results/', 
                        help='Gdzie zapisać wyrenderowane klatki')
    return parser.parse_args()

def get_color(idx):
    """Zwraca unikalny kolor dla danego ID (standard z ByteTrack/YOLOX)."""
    idx = int(idx) * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def extract_frame_id(filepath):
    """Wyciąga numer klatki z nazwy pliku, np. 'd_001.jpg' -> 1"""
    basename = os.path.basename(filepath)
    clean_name = os.path.splitext(basename)[0]
    try:
        return int(clean_name.split('_')[-1])
    except ValueError:
        return 0

def main(args):
    txt_files = sorted(glob.glob(os.path.join(args.track_dir, "*.txt")))
    txt_files = [f for f in txt_files if "meta_data" not in f]

    if not txt_files:
        logger.error(f"Nie znaleziono plików TXT w: {args.track_dir}")
        return

    logger.info(f"Znaleziono {len(txt_files)} sekwencji. Rozpoczynam wizualizację...")

    for txt_path in txt_files:
        seq_name = os.path.splitext(os.path.basename(txt_path))[0]
        
        # 1. Szukanie obrazów z odpowiednim prefiksem sekwencji (np. "d_*.jpg")
        # Zakładamy, że zdjęcia są bezpośrednio w img_root lub w podfolderze 'img'
        img_paths = glob.glob(os.path.join(args.img_root, f"{seq_name}_*.[jp][pn]g"))
        if not img_paths:
            img_paths = glob.glob(os.path.join(args.img_root, "img", f"{seq_name}_*.[jp][pn]g"))
            
        if not img_paths:
            logger.warning(f"Brak zdjęć dla sekwencji '{seq_name}' pasujących do wzorca {seq_name}_*.jpg w {args.img_root}. Pomijam.")
            continue

        # Sortowanie obrazów zgodnie z ich naturalną kolejnością (numeracją)
        img_paths.sort(key=extract_frame_id)

        # Przygotowanie folderu zapisu dla danej sekwencji
        seq_save_dir = os.path.join(args.save_dir, seq_name)
        os.makedirs(seq_save_dir, exist_ok=True)

        # 2. Wczytanie wyników z pliku TXT
        try:
            results = np.loadtxt(txt_path, delimiter=',')
        except Exception as e:
            logger.error(f"Nie udało się wczytać pliku {txt_path}: {e}")
            continue

        if len(results) == 0:
            results = np.empty((0, 10))
        elif results.ndim == 1:
            results = results.reshape(1, -1)
            
        logger.info(f"Renderowanie sekwencji: {seq_name} ({len(img_paths)} klatek)")
        
        # 3. Nakładanie ramek klatka po klatce
        for virtual_frame_id, img_path in tqdm(enumerate(img_paths, start=1), total=len(img_paths), ncols=100):
            
            # W klasycznym formacie MOT detekcje są zindeksowane numerem klatki
            frame_results = results[results[:, 0] == virtual_frame_id]
            
            img = cv2.imread(img_path)
            if img is None:
                continue

            for row in frame_results:
                track_id = int(row[1])
                score = row[6]
                x_min, y_min, w, h = row[2:6]
                
                # Zabezpieczenie na wypadek ujemnych współrzędnych
                x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
                x_max, y_max = int(x_min + w), int(y_min + h)

                color = get_color(track_id)
                
                # Rysowanie ramki
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=2)
                
                # Etykieta tekstu: "ID: score" (lub jak woli domyślny ByteTrack - z samym ID)
                text = f'{track_id}'
                
                # Tło pod tekst dla lepszej czytelności (opcjonalnie, domyślne w track_kalman.py)
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
                cv2.rectangle(img, (x_min, y_min - text_h - 4), (x_min + text_w, y_min), color, -1)
                cv2.putText(img, text, (x_min, y_min - 2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)

            # Zapis wyrenderowanej klatki pod oryginalną nazwą lub narzuconą numeracją
            save_name = os.path.basename(img_path)
            save_path = os.path.join(seq_save_dir, save_name)
            cv2.imwrite(save_path, img)

if __name__ == '__main__':
    args = get_args()
    main(args)