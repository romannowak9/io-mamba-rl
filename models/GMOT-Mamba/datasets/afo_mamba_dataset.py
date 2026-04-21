import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class AFODataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, max_targets=10, num_ref_frames=2):
        self.img_dir = os.path.join(root_dir, split, 'img')
        self.ann_dir = os.path.join(root_dir, split, 'ann')
        self.img_names = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
        self.max_targets = max_targets
        self.num_ref_frames = num_ref_frames

    def __len__(self):
        # Odejmujemy num_ref_frames, bo pierwsza klatka testowa 
        # musi mieć za sobą odpowiednią liczbę klatek referencyjnych
        return len(self.img_names) - self.num_ref_frames

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # Normalizacja do [0, 1] i zmiana układu na [C, H, W]
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    def _parse_json(self, ann_path):
        if not os.path.exists(ann_path):
            return np.zeros((0, 4), dtype=np.float32)
        with open(ann_path, 'r') as f:
            data = json.load(f)
        
        h_orig = data['size']['height']
        w_orig = data['size']['width']
        
        boxes = []
        for obj in data['objects']:
            if obj['geometryType'] == 'rectangle':
                p = obj['points']['exterior']
                x1, y1 = p[0][0] / w_orig, p[0][1] / h_orig
                x2, y2 = p[1][0] / w_orig, p[1][1] / h_orig
                boxes.append([x1, y1, x2, y2])
        
        return np.array(boxes, dtype=np.float32)

    def _pad_boxes(self, boxes):
        final_boxes = np.zeros((self.max_targets, 4), dtype=np.float32)
        n = min(len(boxes), self.max_targets)
        if n > 0:
            final_boxes[:n] = boxes[:n]
        return torch.from_numpy(final_boxes)

    def _generate_gaussian_target(self, boxes, shape=(32, 32)):
        target = np.zeros((self.max_targets, *shape), dtype=np.float32)
        for i, box in enumerate(boxes[:self.max_targets]):
            x1, y1, x2, y2 = box
            ctx = int((x1 + x2) / 2 * shape[1])
            cty = int((y1 + y2) / 2 * shape[0])
            
            for y in range(max(0, cty-2), min(shape[0], cty+3)):
                for x in range(max(0, ctx-2), min(shape[1], ctx+3)):
                    dist = ((x - ctx)**2 + (y - cty)**2)
                    target[i, y, x] = np.exp(-dist / (2 * 1.0**2))
        return torch.from_numpy(target)

    def __getitem__(self, idx):
        # Przesuwamy indeks, aby klatka testowa miała 'num_ref_frames' klatek przed sobą
        test_idx = idx + self.num_ref_frames
        
        # 1. Klatka testowa
        img_next_path = os.path.join(self.img_dir, self.img_names[test_idx])
        ann_next_path = os.path.join(self.ann_dir, self.img_names[test_idx] + ".json")
        img_next = self._load_img(img_next_path)
        boxes_next = self._parse_json(ann_next_path)
        
        # 2. Klatki referencyjne
        ref_frames = []
        ref_boxes = []
        ref_gauss = []
        
        for i in range(test_idx - self.num_ref_frames, test_idx):
            img_path = os.path.join(self.img_dir, self.img_names[i])
            ann_path = os.path.join(self.ann_dir, self.img_names[i] + ".json")
            
            ref_frames.append(self._load_img(img_path))
            boxes = self._parse_json(ann_path)
            ref_boxes.append(self._pad_boxes(boxes))
            ref_gauss.append(self._generate_gaussian_target(boxes, shape=(32, 32)))

        return {
            "train_frames": torch.stack(ref_frames), 
            "test_frame": img_next,
            "train_boxes": torch.stack(ref_boxes),
            "train_gauss": torch.stack(ref_gauss),
            "gt_gauss": self._generate_gaussian_target(boxes_next, (64, 64)),
            "gt_boxes": self._pad_boxes(boxes_next)
        }

def check_dataset_health(root_path):
    print("--- Rozpoczynam sprawdzanie Datasetu AFO (Multi-Frame) ---")
    
    try:
        dataset = AFODataset(root_dir=root_path, split='train', num_ref_frames=3)
        print(f"Znaleziono dostępnych sekwencji: {len(dataset)}")
    except Exception as e:
        print(f"Błąd podczas inicjalizacji: {e}")
        return

    sample = dataset[0]
    
    print("\nKształty tensorów w próbce:")
    for key, value in sample.items():
        print(f"{key:15} -> {value.shape}")

    # Wizualizacja
    img_ref_last = sample['train_frames'][-1].permute(1, 2, 0).numpy() # Ostatnia klatka ref
    img_next = sample['test_frame'].permute(1, 2, 0).numpy()
    
    # Do rysowania boksów potrzebujemy obrazu w formacie uint8 0-255
    img_ref_vis = (img_ref_last * 255).astype(np.uint8).copy()
    
    # Nakładamy boksy z ostatniej klatki referencyjnej
    h, w, _ = img_ref_vis.shape
    for box in sample['train_boxes'][-1]:
        if box.sum() == 0: continue
        x1, y1, x2, y2 = box.numpy()
        cv2.rectangle(img_ref_vis, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img_ref_vis)
    axes[0, 0].set_title("Ostatnia Klatka Ref (t-1) z boksami")
    
    axes[0, 1].imshow(img_next)
    axes[0, 1].set_title("Klatka Testowa (t)")
    
    # Sumujemy heatmapy po wszystkich 10 targetach dla czytelności podglądu
    gauss_ref_sum = sample['train_gauss'][-1].sum(dim=0).numpy()
    gauss_gt_sum = sample['gt_gauss'].sum(dim=0).numpy()

    axes[1, 0].imshow(gauss_ref_sum, cmap='hot')
    axes[1, 0].set_title("Heatmapa Ref (32x32)")
    
    axes[1, 1].imshow(gauss_gt_sum, cmap='hot')
    axes[1, 1].set_title("Heatmapa Target (64x64)")

    plt.tight_layout()
    output_path = "dataset_sanity_check_multi.png"
    plt.savefig(output_path)
    print(f"\nSukces! Podgląd zapisany w: {output_path}")

if __name__ == "__main__":
    DATA_PATH = "data/afo" 
    check_dataset_health(DATA_PATH)