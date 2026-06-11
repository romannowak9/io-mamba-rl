# To run
# python3 -m tools.display_all_vis

import cv2
import glob
import os
from utils.helpers import sort_key_from_filename

# ==========================================================
# KONFIGURACJA ŚCIEŻEK (na podstawie poprzednich skryptów)
# ==========================================================
SEQ = "r3"  # Wybrana sekwencja

PATHS = {
    "GT":  f"out/gt_vis/train/{SEQ}",     # Ścieżka z wyrenderowanym GT
    "DET": f"out/det_results/afo/train/vis_results/{SEQ}",          # Ścieżka z wyrenderowanymi detekcjami
    "TRK": f"out/track_res/afo/train/vis_results/{SEQ}"         # Ścieżka z wyrenderowanym trackingiem
}

def play_three_windows(paths_dict):
    # Wczytanie i bezpieczne posortowanie plików z każdego folderu
    files_gt = sorted(glob.glob(os.path.join(paths_dict["GT"], "*.*")), key=lambda x : sort_key_from_filename(x.split('/')[-1]))
    files_det = sorted(glob.glob(os.path.join(paths_dict["DET"], "*.*")))
    files_trk = sorted(glob.glob(os.path.join(paths_dict["TRK"], "*.*")))

    # Synchronizacja: bierzemy najmniejszą wspólną długość, żeby skrypt nie szukał klatek-duchów
    min_len = min(len(files_gt), len(files_det), len(files_trk))

    if min_len == 0:
        print("[-] Błąd: Jeden lub więcej folderów jest pustych lub podano złą nazwę sekwencji!")
        return

    # Nazwy okien OpenCV
    win_gt = "1. Ground Truth (GT)"
    win_det = "2. Detections (DET)"
    win_trk = "3. Tracking (TRK)"

    # Inicjalizacja, skalowanie i rozsuwanie okien na pulpicie (x_pos)
    # Okna mają domyślnie szerokość 600px i wysokość 400px
    windows_config = [
        (win_gt, 50),       # Pierwsze okno z lewej
        (win_det, 670),     # Środkowe okno (50 + 600 + margines)
        (win_trk, 1290)     # Prawe okno (670 + 600 + margines)
    ]

    for name, x_pos in windows_config:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600, 400)
        cv2.moveWindow(name, x_pos, 150) # 150 to odległość od górnej krawędzi ekranu

    print(f"[+] Odtwarzam {min_len} klatek synchronicznie w 3 oknach.")
    print("[+] Kliknij na dowolne okno i naciśnij [Q], aby wyjść.")

    for i in range(min_len):
        # Odczyt klatek o tym samym indeksie z każdego folderu
        img_gt = cv2.imread(files_gt[i])
        img_det = cv2.imread(files_det[i])
        img_trk = cv2.imread(files_trk[i])

        # Wyświetlenie obrazu w odpowiadającym mu oknie
        if img_gt is not None: cv2.imshow(win_gt, img_gt)
        if img_det is not None: cv2.imshow(win_det, img_det)
        if img_trk is not None: cv2.imshow(win_trk, img_trk)

        # Szybkość odtwarzania: 50ms opóźnienia (~20 FPS)
        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    play_three_windows(PATHS)