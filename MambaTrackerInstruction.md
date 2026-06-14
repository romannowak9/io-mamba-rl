## 0. Wymagania

Trzeba zainstalować w venv wszystko co jest w `mamba_tracker_reqs.txt`. Mogą być problemy z causal-conv1d==1.4.0
mamba-ssm==2.2.2 - one się tak po prostu nie zainstalują. Na repo https://github.com/JackWoo0831/Mamba_Trackers coś tam jest podlinkowane jak to robić.

Żeby dać wyniki dalej na sieć Iwo, to nie musicie tego uruchamiać - wyniki są spushowane w `out/track_res/afo/` - póki co nie wszystkie - tylko część sekwencji z traina - jak byśmy potrzebowali wszystkich, to mówcie - odpale wszystko i wrzucę wyniki.

Trzeba sobie pobrać zbiór afo za pomocą skryptu utils/afo_download.py oraz wagi yolox_x.pth (lub yolox_m.pth) (z https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0)

I chyba jeszcze żeby to Iwa działało to do folderu models/pretrained/ trzeba dać vggm.pth z https://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth

## 1. Uruchomić detekcję

Trzeba mieć:
- skrypt do generowania detekcji dla konkretnego zbioru danych (można wygenerować na podstawie Mamba-Trackers/tools/gen_det_results.py)
- plik z modelem do detekcji .pth (wagi) i pasujący do niego plik exp 

Dla AFO wygląda to tak:

Dla yolox_x:
```bash
python tools/gen_det_afo.py \
    --split train \
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
    --split train \
    --data_root ./data/afo \
    --exp_file exps/yolox_m_afo.py \
    --model_path weights/yolox_m.pth \
    --save_dir out/det_results_yolox_m/afo/{split} \
    --generate_meta_data \
    --vis \
    --native_size  # Zachowaj oryginalny rozmiar obrazów
```

Dostaniemy plik .txt z detekcjami w folderze out/

## 2. Uruchomić śledzenie

Tutaj lepiej by było skorzystać z track.py z ssm_tracker (bo Mamba), ale że jest problem z pobraniem wag dla tego modelu, to można po prostu uruchomić śledzenia za pomocą filtrów kalmana z Mamba_trackers/kalman_tracker/track.py

For yolox_m detection results:
```bash
python -m tools.track_kalman \
    --det_path out/det_results_yolox_m/afo/train \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res/{dataset_name}/{split} \
    --vis
```

## 3. Poprawa wyników śledzenia za pomocą VOT od Iwo

Trzeba uruchomić skrypt `models/VOT/tracking/refine_afo_tracking.py` - ścieżki do dostsowania w stałych na dole kodu

## 4. Wyniki - wizualizacja

Wizualizacja detekcji - jak się dało wcześniej opcję --vis, to obrazy z bouding boxami są w wynikach w podfolderze vis_results/

Wizualizacja gt dla śledzenia - generowanie obrazów z bounding boxami
```bash
python tools/track_gt_vis_afo.py --data_root data/afo --save_dir out/gt_vis --split train
```

Wizualizacja wyników śledzenia po poprawie - generowanie obrazów
```bash
python tools/track_res_vis_afo.py \
  --track_dir out/track_res/afo/train/afo_byte_refined/ \
  --img_root data/afo/train/ \
  --save_dir out/refined_track_vis/
```

### Wyświetlenie wyników
Porównanie wyników z detekcji i śledzenia z gt. Konkretny identyfikator sekwencji i ścieżki z zapisanymi obrazami z wynikami można ustawić w kodzie skruptu tools/display_all_vis.py (najpierw trzeba je wygenerować czterema poprzednimi skryptami)
```bash
python3 -m tools.display_all_vis
```


## 5. Wyniki - ewaluacja

Tych skryptów jeszcze nie ma - ale przydałby się coś do obliczania metryk, żeby można było to jakos ocenić

Skrypt do ewaluacja detekcji: `tools/det_eval_afo.py` - ścieżki do ustawienia jako stałe na dole pliku


Śledzenie to nie wiem jak, skoro nie ma odpowiedniego gt

<!-- Skrypt do ewaluacja śledzenia - tutaj większość metryk będzie gówno warta, bo id obiektów są unikalne przy każdej nowej detekcji, zamiast oznaczać ten sam obiekt na kolejnych klatkach - po prostu AFO nie jest zbiorem do śledzenia, tylko do detekcji i przez to nie daje możliwości oceny jakości śledzenia. Jedyne metryki na które można patrzeć to recall, precision i Det-MOTA (czyli to samo co MOTA w ewaluacji detekcji - MOTA bez uwzględniania poprawności śledzenia) -->

## 6. Wnioski

Generalnie coś tam działa i mamy jakieś wyniki detekcji. Minus jest taki, że przez to że nie da się za bardzo pobrać wag modelu do śledzenia Mambowego, to śledzenie odpalam na trackerze z filtrów Kalmana.

Jedyne co tu jest zrobione to po prostu uruchomienie na tym AFO najpierw detekcji za pomocą yolox, a potem śledzenia za pomocą gotowego trackera, działającego dzięki filtrom Kalamana z repo Mamba_Trackers. Jak byśmy chcieli dodać tu cokolwiek swojego to fajnie by było np. albo dotrenować samego heada Mamba_Trackers/ssm_tracker/MambaTrack pod AFO (czego się nie da zrobić bo nie mamy wag + nie ma odpowiednich labeli w gt)wytrenować albo nawet całą sieć (czego bez labeli z id obiektów w gt też się nie da zrobić). Na ten moment nie mamy w tym nic własnego, po prostu użyliśmy gotowych rzeczy do AFO i to nawet nie ma w tym Mamby. Jedno co to naprawdę można zrobić, przy tym zbiorze, to np. dotrenowanie heada yolox zbiorem AFO dla lepszej detekcji.

Jakby się ogarnęło wagi, to możnaby przynajmniej uruchomić śledzenia na AFO na Mambowych modelach, ale potem ciężko by to było porównać, no bo jak ocenić śledzenie, jeśli w gt tego nie ma?

Kod to gówno, sory, był generowany przez Gemini, ale u mnie działa, jak by się chciało coś z tym dalej robić, to lepiej mu wrzucać i prosić o poprawki, albo tworzenie nowych skryptów na podstawie istniejących niż się w to zagłębiać.

Potem to puszczam na tę sieć Iwo i wynik IoU się pogarsza - wspaniale

<!-- TODO -->
Fajnie by było jako detektor zastosować to https://github.com/HZAI-ZJNU/Mamba-YOLO, zamiast zwykłego YOLO.


## 7. Foramt wyników - czyli jak to dalej przetwarzać

Skrypt tools/track_kalman.py tworzy pliki .txt (jeden dla każdej sekwencji wideo) zgodne z formatem MOTChallenge. Każdy wiersz reprezentuje jeden obiekt na pojedynczej klatce w następującym układzie:

`frame_id, target_id, x, y, width, height, score, -1, -1, -1`

Każdy wiersz pliku .txt to pojedyncza detekcja obiektu w danej klatce, zapisana w formacie MOTChallenge:
- frame_id: Numer klatki (indeksowany od 1).
- target_id: Unikalny identyfikator przypisany do konkretnego obiektu (nie zmienia się w kolejnych klatkach).
- x_min, y_min: Współrzędne lewego górnego rogu ramki (bounding boxa).
- width, height: Szerokość oraz wysokość ramki w pikselach.
- score: Próg ufności (confidence score) modelu dla tego obiektu.
- -1, -1, -1: Wartości domyślne (wypełniacze) wymagane przez specyfikację formatu MOT.

Wyniki śledzenia filtrami Kalamana zapisane są w `track_res/afo/train/afo_byte/`, a po poprawie przez VOT w `track_res/afo/train/afo_byte_refined/`

Ewaluacja w out/eval_results/ - dałem tam metryki z detekcji i wynik uruchomienia poprawy śledzenia przez VOT - tam są wartości IoU przed i po VOCie
