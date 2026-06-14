import os
from ultralytics import YOLO
from loguru import logger

def main():
    model_path = "/home/luki10101/projects/Mamba-YOLO/output_dir/afo_project/mambayolo_afo_final_run/weights/best.pt"
    
    # 1. Inicjalizacja modelu za pomocą Ultralytics
    logger.info("🎬 Ładowanie modelu Mamba-YOLO...")
    model = YOLO(model_path)
    
    # Ścieżka do katalogu z oryginalnymi zdjęciami konkretnej sekwencji lub całego podziału
    # Ultralytics potrafi sam przyjąć ścieżkę do folderu ze zdjęciami jako strumień wejściowy!
    source_images = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/test/a"
    
    # 2. Uruchomienie trackingu bezpośrednio przez silnik modelu
    logger.info("🚀 Uruchamianie śledzenia End-to-End...")
    results = model.track(
        source=source_images,      # Źródło: folder ze zdjęciami, plik wideo lub stream
        tracker="bytetrack.yaml",  # Wbudowana konfiguracja ByteTracka (lub ścieżka do Twojego pliku)
        conf=0.3,                  # Twój niski próg detekcji dla słabych obiektów (track_low_thresh)
        iou=0.8,                   # Twój match_thresh dla asocjacji IoU
        imgsz=1280,                # Model sam przeskaluje obraz do inferencji i wróci do oryginału przy zapisie
        save=True,                 # Automatycznie wyrenderuje i zapisze zdjęcia/wideo z ramkami i ID!
        save_txt=True,             # Automatycznie wygeneruje logi MOT w formacie tekstowym
        project="track_results",   # Katalog główny zapisu
        name="mamba_end_to_end"    # Podfolder dla tego uruchomienia
    )
    
    logger.info("✅ Śledzenie zakończone sukcesem!")

if __name__ == '__main__':
    main()