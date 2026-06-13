# Example usage:
# Stały rozmiar:
# python3 tools/gen_det_afo.py --split train --data_root ./data/afo --exp_file ByteTrack/exps/example/mot/yolox_x_mix_det.py --model_path ByteTrack/weights/yolox_x_mix_det.pth --generate_meta_data --vis
#
# Oryginalny rozmiar (dynamiczny):
# python3 tools/gen_det_afo.py --split train --data_root ./data/afo --exp_file ByteTrack/exps/example/mot/yolox_x_mix_det.py --model_path ByteTrack/weights/yolox_x_mix_det.pth --generate_meta_data --vis --native_size

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# Importy z repozytorium YOLOX/Mamba_Trackers
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils.model_utils import fuse_model, get_model_info
from ByteTrack.yolox.utils.boxes import postprocess

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def get_args():
    parser = argparse.ArgumentParser(description="Generowanie detekcji YOLOX dla zbioru AFO")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'valid', 'test'])
    parser.add_argument('--data_root', required=True, type=str, default='data/afo', help='Ścieżka główna do folderu afo')
    parser.add_argument('--exp_file', required=True, type=str, help='Ścieżka do pliku exp YOLOX')
    parser.add_argument('--model_path', required=True, type=str, help='Ścieżka do wag modelu (.pth)')
    
    # Parametry kontroli rozdzielczości wejściowej
    parser.add_argument('--img_size', nargs='+', type=int, default=[800, 1440], help='Rozdzielczość testowa [H, W] (używana tylko, gdy nie wybrano --native_size)')
    parser.add_argument('--native_size', action='store_true', help='Użyj oryginalnego rozmiaru zdjęcia (zaokrąglonego do wielokrotności 32) zamiast sztywnego --img_size')
    
    parser.add_argument('--high_thresh', type=float, default=0.5, help='Próg ufności dla wizualizacji')
    parser.add_argument('--save_dir', type=str, default='out/det_results/afo/{split}', help='Katalog zapisu wyników')
    parser.add_argument('--device', type=str, default='0', help='Karta GPU lub cpu')
    parser.add_argument('--fp16', action='store_true', help='Użyj precyzji fp16')
    parser.add_argument('--vis', action='store_true', help='Wizualizacja detekcji')
    parser.add_argument('--generate_meta_data', action='store_true', help='Generuj meta_data.txt z wymiarami klatek')
    
    return parser.parse_args()

def select_device(device):
    if device == 'cpu':
        logger.info('Używam CPU do inferencji')
    elif ',' in device:
        logger.error('Multi-GPU nie jest obecnie wspierane w tym skrypcie')
    else:
        logger.info(f'Ustawiam GPU: {device}')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), "CUDA nie jest dostępna!"
    
    cuda = device != 'cpu' and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')

def postprocess_yolox(out, num_classes, conf_thresh, img, ori_img):
    # Wywołanie oryginalnego postprocessingu z YOLOX
    out = postprocess(out, num_classes, conf_thresh)[0]
    if out is None: 
        return None

    # Połączenie score'ów (składnik obiektywności * składnik klasyfikacji)
    out[:, 4] *= out[:, 5]
    out[:, 5] = out[:, -1]
    out = out[:, :-1]

    # Skalowanie boksów z powrotem do oryginalnego rozmiaru zdjęcia
    img_size = [img.shape[-2], img.shape[-1]]
    ori_img_size = [ori_img.shape[0], ori_img.shape[1]]
    scale = min(float(img_size[0]) / ori_img_size[0], float(img_size[1]) / ori_img_size[1])
    out[:, :4] /= scale 

    return out

def save_results(folder_name, seq_name, result_dict):
    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(folder_name, f"{seq_name}.txt")
    
    with open(out_path, 'w') as f:
        for frame_id, output in result_dict.items():
            if output is None:
                continue
            for det in output:
                # Format MOT: frame_id, -1, top_left_x, top_left_y, width, height, conf, -1, -1, -1
                f.write(f'{frame_id},-1,{det[0]:.2f},{det[1]:.2f},{det[2]:.2f},{det[3]:.2f},{det[4]:.2f},-1,-1,-1\n')
    logger.info(f'Zapisano detekcje do: {out_path}')

def save_meta_data(folder_name, meta_data):
    os.makedirs(folder_name, exist_ok=True)
    meta_path = os.path.join(folder_name, 'meta_data.txt')
    with open(meta_path, 'w') as f:
        for k, v in meta_data.items():
            line = f"{k},{v[0]},{v[1]}"
            f.write(line + '\n')
    logger.info(f'Zapisano metadane do: {meta_path}')

def plot_img(img, seq_name, frame_id, results, base_save_dir):
    save_dir = os.path.join(base_save_dir, 'vis_results', seq_name)
    os.makedirs(save_dir, exist_ok=True)

    img_ = np.ascontiguousarray(np.copy(img))
    for det in results:
        # det w tym miejscu to już tlwh (top-left width height)
        tlwh, s = det[:4], det[4]
        tlbr = (int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]))
        text = f'{s:.2f}'

        cv2.rectangle(img_, tlbr[:2], tlbr[2:], (0, 255, 0), thickness=2)
        cv2.putText(img_, text, (tlbr[0], max(tlbr[1] - 5, 10)), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                    color=(255, 164, 0), thickness=2)
        
    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)

def main(args):
    # Inicjalizacja konfiguracji eksperymentu YOLOX
    exp = get_exp(args.exp_file, None)

    device = select_device(args.device)
    model = exp.get_model().to(device)

    logger.info(f"Podsumowanie modelu: {get_model_info(model, exp.test_size)}")
    model.eval()

    # Ładowanie wag
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt["model"])
    logger.info("Wagi modelu załadowane pomyślnie.")

    logger.info("Łączenie warstw modelu (Model Fusion)...")
    model = fuse_model(model)

    if args.fp16:
        model = model.half()
    
    # Mapowanie podziałów na strukturę folderów AFO
    split_map = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
    target_split = split_map[args.split]
    
    img_dir = Path(args.data_root) / target_split / "img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Katalog obrazów nie istnieje: {img_dir}")
        
    save_dir = args.save_dir.format(split=args.split)
    
    # 1. GRUPOWANIE: Zbieranie i porządkowanie płaskiej struktury plików AFO w wirtualne sekwencje
    logger.info("Skanowanie i grupowanie klatek w sekwencje wideo...")
    all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    seq_groups = {}
    for img_path in all_images:
        stem = img_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        
        seq_id = "_".join(parts[:-1]) # Np. 'a' lub 'seq_1'
        try:
            frame_idx = int(parts[-1])
        except ValueError:
            continue
            
        if seq_id not in seq_groups:
            seq_groups[seq_id] = []
        seq_groups[seq_id].append((frame_idx, img_path))

    meta_data = {}

    # 2. INFERENCJA: Przetwarzanie sekwencja po sekwencji
    for seq_id, frame_list in seq_groups.items():
        logger.info(f"Rozpoczynam detekcję dla sekwencji: {seq_id} (Liczba klatek: {len(frame_list)})")
        
        # Sortowanie klatek chronologicznie wg ich numerów
        frame_list.sort(key=lambda x: x[0])
        
        det_result_dict = {}
        
        # Pobranie wymiarów z pierwszej klatki sekwencji na potrzeby metadanych
        first_img = cv2.imread(str(frame_list[0][1]))
        meta_data[seq_id] = [first_img.shape[0], first_img.shape[1]] # H, W
        
        # Tracker oczekuje ciągłych identyfikatorów klatek zaczynających się od 1
        virtual_frame_id = 1 

        for frame_idx, img_path in tqdm(frame_list, desc=f"Sekwencja {seq_id}"):
            img_ori = cv2.imread(str(img_path))
            
            # --- POPRAWKA: Dynamiczny wybór rozdzielczości przetwarzania ---
            if args.native_size:
                h_ori, w_ori = img_ori.shape[:2]
                current_img_size = (int(np.ceil(h_ori / 32) * 32), int(np.ceil(w_ori / 32) * 32))
            else:
                current_img_size = args.img_size
            # ---------------------------------------------------------------
            
            # Preprocessing z użyciem wybranego rozmiaru
            img, ratio = preproc(img_ori, current_img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            if args.fp16:
                img = img.half()

            with torch.no_grad():
                output = model(img)
                # Wyjściowy format z postprocess_yolox to [N, 5] -> (x1, y1, x2, y2, conf)
                output = postprocess_yolox(output, exp.num_classes, 0.05, img, img_ori)  

            if output is not None:
                # Konwersja współrzędnych z formatu tlbr (x1, y1, x2, y2) na format tlwh (x1, y1, w, h)
                output[:, 2] -= output[:, 0]
                output[:, 3] -= output[:, 1]
                
                # Przeniesienie wyników na procesor
                output_np = output.cpu().numpy()
            else:
                output_np = np.empty((0, 5))

            if args.vis and output_np.shape[0] > 0:
                # Filtrowanie do wizualizacji za pomocą progu high_thresh
                vis_output = output_np[output_np[:, 4] > args.high_thresh]
                plot_img(img_ori, seq_id, virtual_frame_id, vis_output, save_dir)

            det_result_dict[virtual_frame_id] = output_np
            virtual_frame_id += 1

        # Zapisz plik tekstowy z detekcjami dla danej sekwencji
        save_results(save_dir, seq_id, det_result_dict)

    # Zapis zbiorczego pliku metadanych, jeśli flaga jest aktywna
    if args.generate_meta_data:
        save_meta_data(save_dir, meta_data)


if __name__ == '__main__':
    args = get_args()
    main(args)