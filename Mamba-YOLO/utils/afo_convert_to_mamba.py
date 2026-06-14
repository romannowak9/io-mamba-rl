import json
import os
from pathlib import Path

# 1. Definicja mapowania klas na podstawie przesłanych metadanych
# YOLO wymaga indeksów numerycznych zaczynających się od 0
CLASS_MAPPING = {
    "boat": 0,
    "bouy": 1,
    "human": 2,
    "kayak": 3,
    "large_obj": 4,
    "object": 5,
    "sailboat": 6,
    "small_obj": 7,
    "wind/sup-board": 8
}

def convert_single_json(json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Błąd dekodowania pliku JSON: {json_path}")
            return

    # Pobranie wymiarów obrazu potrzebnych do normalizacji
    img_w = data.get("size", {}).get("width")
    img_h = data.get("size", {}).get("height")
    
    if not img_w or not img_h:
        print(f"⚠️ Pominięto {json_path.name} - brak informacji o rozmiarze obrazu.")
        return

    yolo_lines = []

    for obj in data.get("objects", []):
        class_title = obj.get("classTitle")
        
        # Sprawdzamy, czy dana klasa znajduje się w naszym mapowaniu
        if class_title not in CLASS_MAPPING:
            continue
            
        class_id = CLASS_MAPPING[class_title]
        
        # Pobranie punktów geometrii prostokąta
        points = obj.get("points", {}).get("exterior", [])
        if len(points) < 2:
            continue
            
        x_min, y_min = points[0][0], points[0][1]
        x_max, y_max = points[1][0], points[1][1]
        
        # Obliczenie szerokości (W), wysokości (H) oraz współrzędnych środka (X, Y)
        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + (w / 2.0)
        y_center = y_min + (h / 2.0)
        
        # Normalizacja wartości do zakresu 0.0 - 1.0
        x_norm = x_center / img_w
        y_norm = y_center / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        # Ograniczamy zapis do 6 miejsc po przecinku w celu zachowania czystości pliku
        yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Określenie nazwy pliku wyjściowego (zamiana rozszerzeń np. a_102.jpg.json -> a_102.txt)
    # Odcinamy '.json', a jeśli nazwa to 'a_102.jpg.json', zamieni się w 'a_102.txt'
    base_name = json_path.name
    if base_name.endswith('.jpg.json'):
        out_name = base_name.replace('.jpg.json', '.txt')
    elif base_name.endswith('.png.json'):
        out_name = base_name.replace('.png.json', '.txt')
    else:
        out_name = json_path.stem + '.txt'

    output_file_path = Path(output_dir) / out_name
    
    # Zapis tylko wtedy, gdy w pliku znajdują się jakieś adnotacje
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(yolo_lines))


def batch_convert(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Tworzenie folderu wyjściowego, jeśli nie istnieje
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Wyszukanie wszystkich plików json w katalogu
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"Brak plików .json w katalogu: {input_folder}")
        return
        
    print(f"🚀 Rozpoczynam konwersję {len(json_files)} plików JSON...")
    
    converted_count = 0
    for j_file in json_files:
        # Pomijamy plik meta.json (plik z definicją klas, który mi podesłałeś)
        if j_file.name == "meta.json":
            continue
            
        convert_single_json(j_file, output_path)
        converted_count += 1
        
    print(f"✅ Zakończono! Przetworzono pomyślnie {converted_count} plików.")


if __name__ == "__main__":
    # --- TUTAJ WPISZ SWOJE ŚCIEŻKI ---
    INPUT_DIR = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/test/ann"       # Folder z Twoimi plikami .json
    OUTPUT_DIR = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/test/labels"   # Gdzie mają zapisać się pliki .txt dla YOLO
    
    batch_convert(INPUT_DIR, OUTPUT_DIR)