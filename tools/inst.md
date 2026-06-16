# Uruchomianie poszczególnych modułów

## 0. Wymagania

Pobrać zbiór AFO za pomocą skryptu utils/afo_download.py
Pobrać wagi modelu przeznaczonego do detekcji: yolox_x.pth (lub yolox_m.pth) (z https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) lub wagi przetrenowanego Mamba-YOLO (plik best.pt) i zapisać je np. w folderze weights/
Pobrać wagi modelu VOT (plik rl_training_best_afo_1.pt)

Sklonować do głównego katalogu projektu:
 - https://github.com/JackWoo0831/Mamba_Trackers
 - https://github.com/FoundationVision/ByteTrack
 - https://github.com/HZAI-ZJNU/Mamba-YOLO

I zainstalować narzędzia zawarte w sklonowanych repozytoriach zgodnie z instrukcjami z odpowiednich Readme.

Zainstalować wymagania z `requirements.txt`

## 1. Uruchomić detekcję

Wygenererować wyniki detekcji za pomocą jednego z modeli

Dla yolox_x:
```bash
python tools/gen_det_afo.py \
    --split test \
    --data_root ./data/afo \
    --exp_file exps/yolox_x_afo.py \
    --model_path weights/yolox_x.pth \
    --save_dir out/det_results_yolox_x/afo/{split} \
    --generate_meta_data \
    --vis \
    --native_size  # Zachowaj oryginalny rozmiar obrazów
```

Dla yolox_m:
```bash
python tools/gen_det_afo.py \
    --split test \
    --data_root ./data/afo \
    --exp_file exps/yolox_m_afo.py \
    --model_path weights/yolox_m.pth \
    --save_dir out/det_results_yolox_m/afo/{split} \
    --generate_meta_data \
    --vis \
    --native_size  # Zachowaj oryginalny rozmiar obrazów
```

Dla mamba-yolo:
```bash
python tools/gen_det_afo_mamba.py \
    --split test \
    --data_root ./data/afo \
    --model_path ./weights/best_two.pt \
    --save_dir ./out/det_results_mamba/afo/{split} \
    --generate_meta_data \
    --vis \
    --imgsz 960 \
    --conf 0.05
```

Wyjściem są pliki .txt z detekcjami dla każdej sekwencji pod ścieżką --save_dir

Wyniki detekcji można ocenić za pomocą skryptu `models/VOT/tracking/refine_afo_tracking.py`, po odpowiednim dostosowaniu ścieżek do plików wyjściowych z detekcji (stałe w kodzie skryptu)

## 2. Uruchomić śledzenie

Następnie na podstawie wyników detekcji, niezależnie od sposobu ich uzyskania, uruchamiamy algorytm śledzenia

Śledzenie za pomocą algorytmu opartym na bytetrack dla wyników detekcji z yolo_m:
```bash
python -m tools.track_kalman \
    --det_path out/det_results_yolox_m/afo/test \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res/{dataset_name}/{split} \
    --vis
```
Śledzenie za pomocą algorytmu opartym na bytetrack dla wyników detekcji z mamba-yolo:
```bash
python -m tools.track_kalman \
    --det_path out/det_results_mamba/afo/test \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res_mamba/{dataset_name}/{split} \
    --vis
```

Wyjściem są pliki .txt z wynikami śledzenia dla każdej sekwencji pod ścieżką --save_dir

### Format wyjścia

Skrypt tools/track_kalman.py tworzy pliki .txt (jeden dla każdej sekwencji wideo) zgodne z formatem MOTChallenge - taki format był stosowany przez twórców projektu Mamba_Trackers. Każdy wiersz reprezentuje jeden obiekt na pojedynczej klatce w następującym układzie:

`frame_id, target_id, x, y, width, height, score, -1, -1, -1`

Każdy wiersz pliku .txt to pojedyncza detekcja obiektu w danej klatce, zapisana w formacie MOTChallenge:
- frame_id: Numer klatki (indeksowany od 1).
- target_id: Unikalny identyfikator przypisany do konkretnego obiektu (nie zmienia się w kolejnych klatkach).
- x_min, y_min: Współrzędne lewego górnego rogu ramki (bounding boxa).
- width, height: Szerokość oraz wysokość ramki w pikselach.
- score: Próg ufności (confidence score) modelu dla tego obiektu.
- -1, -1, -1: Wartości domyślne (wypełniacze) wymagane przez specyfikację formatu MOT.

Wyniki śledzenia filtrami Kalamana zapisane są w `track_res/afo/test/afo_byte/`, a po poprawie przez VOT w `track_res/afo/test/afo_byte_refined/`

Ewaluacja w out/eval_results/ - dałem tam metryki z detekcji i wynik uruchomienia poprawy śledzenia przez VOT - tam są wartości IoU przed i po VOCie

## 3. Poprawa wyników śledzenia za pomocą VOT

Uruchomić skrypt `models/VOT/tracking/refine_afo_tracking.py` - ścieżki do danych wejściowych i wyjściowych do dostosowania jako stałe w kodzie skryptu

Wyjściem są pliki .txt z wynikami śledzenia dla każdej sekwencji o formacie identycznym jak pliki wyjściowe z poprzedniego modułu (śledzenie filtrami Kalmana).

## 4. Wyniki - wizualizacja

Wizualizacja detekcji - jeżeli przy generowaniu detekcji wybrało się opcję --vis, to obrazy z bouding boxami są w wynikach w podfolderze vis_results/

Wizualizacja groundthruth dla śledzenia - generowanie obrazów z bounding boxami
```bash
python tools/track_gt_vis_afo.py --data_root data/afo --save_dir out/gt_vis --split test
```

Wizualizacja wyników śledzenia po przpuszczeniu przez VOT - generowanie obrazów
```bash
python tools/track_res_vis_afo.py \
  --track_dir out/track_res/afo/test/afo_byte_refined/ \
  --img_root data/afo/test/ \
  --save_dir out/refined_track_vis/
```

### Wyświetlenie wyników

Jeśli wszystkie obrazy z poprzedniego kroku zostały wygenerowane, to można wyświetlić wyniki dla wybranej sekwencji za pomocą skryptu

```bash
python3 -m tools.display_all_vis
```

Konkretny identyfikator sekwencji i ścieżki z zapisanymi obrazami z wynikami można ustawić w kodzie skruptu `tools/display_all_vis.py`
