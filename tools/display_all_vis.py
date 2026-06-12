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
    "GT":      f"out/gt_vis/train/{SEQ}",                             # Ścieżka z wyrenderowanym GT
    "DET":     f"out/det_results_yolox_m/afo/train/vis_results/{SEQ}", # Ścieżka z wyrenderowanymi detekcjami
    "TRK":     f"out/track_res/afo/train/vis_results/{SEQ}",          # Ścieżka z wyrenderowanym trackingiem (np. ByteTrack)
    "TRK_VOT": f"out/refined_track_vis/{SEQ}"                           # Ścieżka z wyrenderowanym TRK_VOT
}

def play_four_windows(paths_dict):
    # Wczytanie i bezpieczne posortowanie plików z każdego folderu
    files_gt = sorted(glob.glob(os.path.join(paths_dict["GT"], "*.*")), key=lambda x : sort_key_from_filename(x.split('/')[-1]))
    files_det = sorted(glob.glob(os.path.join(paths_dict["DET"], "*.*")))
    files_trk = sorted(glob.glob(os.path.join(paths_dict["TRK"], "*.*")))
    files_vot = sorted(glob.glob(os.path.join(paths_dict["TRK_VOT"], "*.*")), key=lambda x : sort_key_from_filename(x.split('/')[-1]))

    # Synchronizacja: bierzemy najmniejszą wspólną długość z 4 folderów
    min_len = min(len(files_gt), len(files_det), len(files_trk), len(files_vot))

    if min_len == 0:
        print("[-] Błąd: Jeden lub więcej folderów jest pustych lub podano złą nazwę sekwencji!")
        print(f"Liczba plików: GT={len(files_gt)}, DET={len(files_det)}, TRK={len(files_trk)}, TRK_VOT={len(files_vot)}")
        return

    # Nazwy okien OpenCV
    win_gt = "1. Ground Truth (GT)"
    win_det = "2. Detections (DET)"
    win_trk = "3. Tracking (TRK)"
    win_vot = "4. Tracking VOT (TRK_VOT)"

    # Inicjalizacja, skalowanie i rozsuwanie okien na pulpicie (x_pos)
    # Zmniejszyłem szerokość okien do 450px, żeby 4 okna zmieściły się obok siebie w linii (4 * 450px = 1800px)
    win_width = 450
    win_height = 320
    
    windows_config = [
        (win_gt, 10),                            # 1. okno
        (win_det, 10 + (win_width + 15) * 1),    # 2. okno
        (win_trk, 10 + (win_width + 15) * 2),    # 3. okno
        (win_vot, 10 + (win_width + 15) * 3)     # 4. okno (TRK_VOT)
    ]

    for name, x_pos in windows_config:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, win_width, win_height)
        cv2.moveWindow(name, x_pos, 150) # 150 to odległość od górnej krawędzi ekranu

    print(f"[+] Odtwarzam {min_len} klatek synchronicznie w 4 oknach.")
    print("[+] Kliknij na dowolne okno i naciśnij [Q], aby wyjść.")

    for i in range(min_len):
        # Odczyt klatek o tym samym indeksie z każdego folderu
        img_gt = cv2.imread(files_gt[i])
        img_det = cv2.imread(files_det[i])
        img_trk = cv2.imread(files_trk[i])
        img_vot = cv2.imread(files_vot[i])

        # Wyświetlenie obrazu w odpowiadającym mu oknie
        if img_gt is not None: cv2.imshow(win_gt, img_gt)
        if img_det is not None: cv2.imshow(win_det, img_det)
        if img_trk is not None: cv2.imshow(win_trk, img_trk)
        if img_vot is not None: cv2.imshow(win_vot, img_vot)

        # Szybkość odtwarzania: 50ms opóźnienia (~20 FPS)
        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    play_four_windows(PATHS)