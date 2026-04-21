
import torch
from model import GMOTMamba
from datasets.afo_mamba_dataset import AFODataset
import pytorch_lightning as pl
from gmot_lightning import GMOTMambaTrainer

def test_model():
    # 1. Konfiguracja
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")
    
    batch_size = 2
    m_targets = 10
    img_size = 512 # Zgodnie z Twoim założeniem dla WFPLayer (16x16 feat map)
    
    # 2. Inicjalizacja modelu
    model = GMOTMamba(m_targets=m_targets, dim=512).to(device)
    model.eval() # Tryb ewaluacji (wyłącza Dropout/BatchNorm)

    # 3. Tworzenie sztucznych danych wejściowych (Dummy Data)
    # Klatki: [B, 3, H, W]
    train_frames = torch.randn(batch_size, 3, img_size, img_size).to(device)
    test_frame = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Boxy: [B, m, 4] (L, T, R, B)
    train_boxes = torch.rand(batch_size, m_targets, 4).to(device)
    
    # Mapy Gaussa: [B, m, 32, 32] -> Zakładając, że res4 ma 32x32
    # Uwaga: Musisz upewnić się, czy TargetStateEncoding oczekuje map 32x32 czy 16x16
    train_gauss = torch.randn(batch_size, m_targets, 32, 32).to(device)

    print("Starting forward pass...")
    
    try:
        with torch.no_grad():
            output = model(train_frames, test_frame, train_boxes, train_gauss)
        
        print("\nSuccess! Output shapes:")
        print(f"Classification map: {output['cls_score'].shape}") # Spodziewane: [2, 10, 64, 64] (zależnie od PFF)
        print(f"Bounding boxes:    {output['bbox_coords'].shape}") # Spodziewane: [2, 10, 4, 64, 64]
        
    except Exception as e:
        print("\nForward pass failed!")
        print(f"Error: {e}")

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

def train_gmot():
    # 1. Konfiguracja danych
    train_dataset = AFODataset(root_dir="data/afo", split='train', max_targets=6, num_ref_frames=3)
    val_dataset = AFODataset(root_dir="data/afo", split='validation', max_targets=6, num_ref_frames=3)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 2. Inicjalizacja modelu rdzeniowego i trenera Lightning
    model_core = GMOTMamba(m_targets=6, dim=512)
    lightning_model = GMOTMambaTrainer(model=model_core, lr=1e-4)

    # 3. Callbacks i Loggery
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="gmot-mamba-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    logger = TensorBoardLogger("tb_logs", name="gmot_mamba_afo")

    # 4. Trainer - tutaj dzieje się magia
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu", # Użyj "gpu" dla WSL z CUDA
        devices=1,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        precision="16-mixed", # Szybszy trening dzięki automatycznej precyzji mieszanej
    )

    # 5. START!
    trainer.fit(lightning_model, train_loader, val_loader)

if __name__ == "__main__":
    train_gmot()
    # test_model()