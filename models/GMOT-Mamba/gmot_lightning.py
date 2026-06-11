import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection import MeanAveragePrecision

class GMOTMambaTrainer(pl.LightningModule):
    def __init__(self, model, lr=1e-4, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # Osobne metryki dla Treningu i Walidacji
        self.train_map_metric = MeanAveragePrecision(box_format="xyxy")
        self.val_map_metric = MeanAveragePrecision(box_format="xyxy")

    def forward(self, train_frames, test_frame, train_boxes, train_gauss):
        return self.model(train_frames, test_frame, train_boxes, train_gauss)

    def _compute_loss(self, batch, outputs):
        # 1. Classification Loss (MSE na heatmapach)
        pred_cls = torch.sigmoid(outputs['cls_score'])
        loss_cls = F.mse_loss(pred_cls, batch['gt_gauss'])

        # 2. Regression Loss (L1) z maskowaniem
        target_boxes = batch['gt_boxes'].unsqueeze(-1).unsqueeze(-1).expand_as(outputs['bbox_coords'])
        mask = batch['gt_gauss'].unsqueeze(2) > 0.1 
        
        if mask.any():
            loss_reg = F.l1_loss(outputs['bbox_coords'][mask.expand_as(outputs['bbox_coords'])], 
                                 target_boxes[mask.expand_as(outputs['bbox_coords'])])
        else:
            loss_reg = 0.0

        total_loss = loss_cls + 1.0 * loss_reg
        return total_loss, loss_cls, loss_reg

    def training_step(self, batch, batch_idx):
        outputs = self(batch['train_frames'], batch['test_frame'], 
                       batch['train_boxes'], batch['train_gauss'])
        
        loss, l_cls, l_reg = self._compute_loss(batch, outputs)
        
        # --- DEKODOWANIE DLA KAŻDEJ ITERACJI ---
        preds = self._decode_batch(outputs)
        targets = self._prepare_targets(batch)
        
        # Aktualizacja metryki
        self.train_map_metric.update(preds, targets)
        print("TARGETS:", targets)
        print("PREDS:", preds)
        
        # Obliczamy mAP tylko dla tego konkretnego kroku (batcha)
        # Używamy clone().compute(), aby nie resetować globalnego licznika epoki
        step_map = self.train_map_metric.compute()
        
        # Logowanie do wykresu "step"
        self.log("train/loss_step", loss, prog_bar=True, on_step=True)
        self.log("train/mAP_step", step_map["map"], on_step=True, prog_bar=True)
        self.log("train/mAP_50_step", step_map["map_50"], on_step=True)
        
        # Ważne: czyścimy metrykę per-step, jeśli chcemy mieć "czysty" wynik kroku,
        # ale zazwyczaj w Lightning lepiej pozwolić jej się akumulować do końca epoki
        # i logować wynik skumulowany. Jeśli chcesz wynik TYLKO z tego batcha:
        # self.train_map_metric.reset() 

        return loss

    def on_train_epoch_end(self):
        # Obliczenie i logowanie mAP dla treningu
        mAP_results = self.train_map_metric.compute()
        self.log("train/mAP", mAP_results["map"], prog_bar=True)
        self.log("train/mAP_50", mAP_results["map_50"])
        self.train_map_metric.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['train_frames'], batch['test_frame'], 
                       batch['train_boxes'], batch['train_gauss'])
        
        loss, _, _ = self._compute_loss(batch, outputs)
        
        preds = self._decode_batch(outputs)
        targets = self._prepare_targets(batch)
        
        self.val_map_metric.update(preds, targets)
        
        # Logujemy mAP dla każdego kroku walidacji
        val_step_map = self.val_map_metric.compute()
        self.log("val/mAP_step", val_step_map["map"], on_step=True)
        
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        mAP_results = self.val_map_metric.compute()
        self.log("val/mAP", mAP_results["map"], prog_bar=True)
        self.log("val/mAP_50", mAP_results["map_50"], prog_bar=True)
        self.val_map_metric.reset()

    def _decode_batch(self, outputs):
        B, m, _, H, W = outputs['bbox_coords'].shape
        # Używamy sigmoid, aby upewnić się, że ufność jest 0-1
        cls_scores = torch.sigmoid(outputs['cls_score']) 
        
        # Bardzo ważne: Czy regresja boksów ma sigmoid? 
        # Jeśli nie, wartości mogą być poza [0, 1]. 
        # Załóżmy bezpiecznie, że chcemy je ograniczyć do obrazu.
        reg_coords = torch.sigmoid(outputs['bbox_coords']) 
        
        decoded_preds = []
        for b in range(B):
            boxes, scores, labels = [], [], []
            for obj_idx in range(m):
                score_map = cls_scores[b, obj_idx]
                max_val, max_idx = torch.max(score_map.view(-1), dim=0)
                
                # Obniż próg do debugowania, żeby zobaczyć czy cokolwiek "żyje"
                if max_val < 0.01: 
                    continue
                
                y_idx = max_idx // W
                x_idx = max_idx % W
                
                # Wyciągamy boks i skalujemy do rozmiaru obrazu (512)
                # Format: [x1, y1, x2, y2]
                box = reg_coords[b, obj_idx, :, y_idx, x_idx] * 512 # 512
                
                # WALIDACJA GEOMETRII: x2 musi być > x1, y2 > y1
                # Jeśli model się uczy, czasem te wartości są zamienione.
                x1, y1, x2, y2 = box
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                
                boxes.append(torch.stack([x1, y1, x2, y2]))
                scores.append(max_val)
                labels.append(torch.tensor(0, device=self.device))

            if len(boxes) > 0:
                decoded_preds.append({
                    "boxes": torch.stack(boxes).detach(),
                    "scores": torch.stack(scores).detach(),
                    "labels": torch.stack(labels).detach()
                })
            else:
                decoded_preds.append({
                    "boxes": torch.empty((0, 4), device=self.device),
                    "scores": torch.empty((0,), device=self.device),
                    "labels": torch.empty((0,), dtype=torch.long, device=self.device)
                })
        return decoded_preds

    def _prepare_targets(self, batch):
        targets = []
        for b in range(batch['gt_boxes'].shape[0]):
            valid_mask = batch['gt_boxes'][b].sum(dim=1) != 0
            # MNOŻYMY PRZEZ 512, aby dopasować do skali boksów z _decode_batch
            boxes = batch['gt_boxes'][b][valid_mask] * 512 
            targets.append({
                "boxes": boxes.to(self.device),
                "labels": torch.zeros(len(boxes), dtype=torch.long, device=self.device)
            })
        return targets

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}