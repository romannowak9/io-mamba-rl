# Usage: python3 tools/gen_traj_afo.py --split train --save_name afo_train
import json
import argparse
from pathlib import Path

# Definicje zestawów klas dla zbioru AFO
CLASS_MODES = {
    1: ["object"],
    2: ["small_obj", "large_obj"],
    6: ["human", "surfboard", "boat", "bird", "drone", "ufo"]
}

def gen_afo_trajectory_data(data_root, split="train", sequence_filter=None, filter_len=1, obj_id_base=0, class_mode=1):
    print(f"Rozpoczynanie parsowania AFO | Podział: {split} | Tryb klas: {class_mode}-class")
    
    # Walidacja wybranego trybu klas
    if class_mode not in CLASS_MODES:
        raise ValueError(f"Nieprawidłowy class_mode. Wybierz spośród: {list(CLASS_MODES.keys())}")
    allowed_classes = CLASS_MODES[class_mode]
    
    split_map = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
    target_split = split_map[split]
    
    ann_dir = Path(data_root) / "afo" / target_split / "ann"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Katalog adnotacji nie istnieje: {ann_dir}")
        
    raw_trajectories = {}
    json_files = list(ann_dir.glob("*.json"))
    print(f"Znaleziono {len(json_files)} plików adnotacji.")
    
    ignored_count = 0
    accepted_count = 0
    
    for json_path in json_files:
        filename = json_path.name
        clean_stem = filename.replace(".jpg.json", "").replace(".json", "")
        
        parts = clean_stem.split("_")
        if len(parts) < 2:
            continue
            
        seq_id = "_".join(parts[:-1])
        
        # Filtrowanie po konkretnej sekwencji (jeśli podano)
        if sequence_filter and seq_id != sequence_filter:
            continue
            
        try:
            frame_idx = int(parts[-1])
        except ValueError:
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)
            
        image_h = int(data["size"]["height"])
        image_w = int(data["size"]["width"])
        
        for obj in data.get("objects", []):
            class_title = obj.get("classTitle")
            
            # KLUCZOWA ZMIANA: Filtrujemy bboxy przypisane do wybranego trybu klas
            if class_title not in allowed_classes:
                ignored_count += 1
                continue
                
            obj_id = obj.get("id")
            exterior = obj.get("points", {}).get("exterior", None)
            
            if obj_id is None or not exterior or len(exterior) < 2:
                continue
                
            accepted_count += 1
            
            # Wyliczanie współrzędnych
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            w_px = float(x_max - x_min)
            h_px = float(y_max - y_min)
            x_tl = float(x_min)
            y_tl = float(y_min)
            
            # Format oczekiwany przez Mambę: [cx, cy, w, h] znormalizowany do 1.0
            cx_px = x_tl + 0.5 * w_px
            cy_px = y_tl + 0.5 * h_px
            
            cx = round(cx_px / float(image_w), 6)
            cy = round(cy_px / float(image_h), 6)
            w = round(w_px / float(image_w), 6)
            h = round(h_px / float(image_h), 6)
            
            bbox = [cx, cy, w, h]
            
            track_key = (seq_id, obj_id)
            if track_key not in raw_trajectories:
                raw_trajectories[track_key] = []
                
            raw_trajectories[track_key].append((frame_idx, bbox, image_h, image_w))
            
    print(f"Filtrowanie boksów zakońone. Zaakceptowano: {accepted_count}, Odrzucono duplikatów: {ignored_count}")
    
    # Budowanie struktury DanceTrack-style
    annotation_dataset = {}
    new_obj_id = 0
    
    for (seq_id, obj_id), appearances in raw_trajectories.items():
        appearances.sort(key=lambda x: x[0])
        
        traj_len = len(appearances)
        if traj_len < filter_len:
            continue
            
        bboxes = [x[1] for x in appearances]
        img_h = appearances[0][2]
        img_w = appearances[0][3]
        
        annotation_dataset[str(new_obj_id + obj_id_base)] = {
            "image_h": img_h,
            "image_w": img_w,
            "traj_len": traj_len,
            "bboxes": bboxes
        }
        new_obj_id += 1
        
    print(f"Wygenerowano pomyślnie {new_obj_id} unikalnych trajektorii.")
    
    annotation_dataset["total_objs"] = new_obj_id
    annotation_dataset["obj_id_start"] = obj_id_base
    
    return annotation_dataset, obj_id_base + new_obj_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generowanie unikalnych trajektorii AFO z filtrem klasowym")
    parser.add_argument("--data_dir", type=str, default="data", help="Ścieżka do katalogu głównego")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "valid", "test"])
    parser.add_argument("--sequence", type=str, default=None, help="Opcjonalne filtrowanie po nazwie sekwencji")
    parser.add_argument("--filter_len", type=int, default=1, help="Minimalna długość trajektorii (użyj 1 dla AFO)")
    parser.add_argument("--class_mode", type=int, default=6, choices=[1, 2, 6], help="Wybór poziomu klasyfikacji (1, 2 lub 6 klas)")
    parser.add_argument("--save_name", type=str, default="afo_train_single_class", help="Nazwa pliku wynikowego")
    args = parser.parse_args()
    
    annotation_all = {}
    
    afo_data, _ = gen_afo_trajectory_data(
        data_root=args.data_dir,
        split=args.split,
        sequence_filter=args.sequence,
        filter_len=args.filter_len,
        obj_id_base=0,
        class_mode=args.class_mode
    )
    
    annotation_all["afo"] = afo_data
    
    output_dir = Path("./out/traj_anno_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{args.save_name}_{args.class_mode}_classes{('_' + args.sequence) if args.sequence is not None else ''}.json"
    with open(output_file, "w") as f:
        json.dump(annotation_all, f, indent=4)
        
    print(f"Zapisano czysty plik konfiguracyjny w: {output_file}")
