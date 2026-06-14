import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from ultralytics import YOLO

# 1. Zabezpieczenie dla PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

# Odwrócone mapowanie klas (indeks -> nazwa tekstowa) do celów rysowania
CLASS_NAMES = {
    0: "boat", 1: "bouy", 2: "human", 3: "kayak", 4: "large_obj",
    5: "object", 6: "sailboat", 7: "small_obj", 8: "wind/sup-board"
}

def load_ground_truth(txt_path, img_w, img_h):
    """Wczytuje znormalizowane ramki YOLO i przekształca na piksele [x_min, y_min, w, h]"""
    gt_boxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    # Format YOLO: x_center, y_center, width, height (znormalizowane)
                    x_c, y_c, w, h = map(float, parts[1:])
                    
                    # Konwersja na piksele i format dla matplotlib (lewy górny róg + w + h)
                    abs_w = w * img_w
                    abs_h = h * img_h
                    abs_x = (x_c - w / 2.0) * img_w
                    abs_y = (y_c - h / 2.0) * img_h
                    
                    gt_boxes.append({"cls": cls_id, "bbox": [abs_x, abs_y, abs_w, abs_h]})
    except FileNotFoundError:
        print(f"⚠️ Nie znaleziono pliku referencyjnego: {txt_path}")
    return gt_boxes

def run_comparison():
    # --- ZDEFINIUJ SWOJE ŚCIEŻKI ---
    model_path = "/home/luki10101/projects/Mamba-YOLO/output_dir/afo_project/mambayolo_afo_final_run/weights/best.pt"
    image_path = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/train/images/a_102.jpg"
    gt_path = "/home/luki10101/projects/Mamba-YOLO/datasets/afo/train/labels/a_102.txt"

    # 2. Wczytanie obrazu i wymiarów przez OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 3. Pobranie ramek Ground Truth
    gt_boxes = load_ground_truth(gt_path, w, h)

    # 4. Inferencja modelem Mamba-YOLO
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=1280, device=0, half=True)
    print("\nPredykcje modelu Mamba-YOLO:\n", results[0])
    
    pred_boxes = []
    for result in results:
        for box in result.boxes:
            # Format xyxy z ultralytics: [x1, y1, x2, y2] w pikselach
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            # Konwersja na format matplotlib: [x_min, y_min, w, h]
            abs_w = x2 - x1
            abs_h = y2 - y1
            pred_boxes.append({"cls": cls_id, "conf": conf, "bbox": [x1, y1, abs_w, abs_h]})

    # 5. Rysowanie wykresu porównawczego za pomocą Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Wykres 1: Ground Truth (Referencja)
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Etykiety Referencyjne (Ground Truth) | Obiektów: {len(gt_boxes)}", fontsize=14, color='green')
    axes[0].axis('off')
    
    for gt in gt_boxes:
        bx, by, bw, bh = gt["bbox"]
        # Zielona ramka dla danych prawdziwych
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='g', facecolor='none')
        axes[0].add_patch(rect)
        label = CLASS_NAMES.get(gt["cls"], str(gt["cls"]))
        axes[0].text(bx, by - 10, label, color='g', fontsize=10, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Wykres 2: Mamba-YOLO Predictions (Predykcja)
    axes[1].imshow(img_rgb)
    axes[1].set_title(f"Predykcja Mamba-YOLO | Obiektów: {len(pred_boxes)}", fontsize=14, color='blue')
    axes[1].axis('off')
    
    for pred in pred_boxes:
        bx, by, bw, bh = pred["bbox"]
        # Czerwona/Niebieska ramka dla predykcji modelu
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        label = f"{CLASS_NAMES.get(pred['cls'], str(pred['cls']))} {pred['conf']:.2f}"
        axes[1].text(bx, by - 10, label, color='r', fontsize=10, weight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

    plt.tight_layout()
    
    # Zapisanie wykresu do pliku graficznego
    output_plot_path = "porownanie_detekcji.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Wykres porównawczy został pomyślnie zapisany jako: {output_plot_path}")
    
    # Wyświetlenie okna z wykresem (jeśli system posiada interfejs graficzny)
    plt.show()

if __name__ == "__main__":
    run_comparison()