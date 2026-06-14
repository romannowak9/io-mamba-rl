import sys
import os
import torch
import yaml

sys.path.append(os.path.abspath('./Mamba_Trackers'))  # Dodajemy ścieżkę do katalogu z modelem

#from ssm_tracker.models.MambaTrack import MambaTrack 

from Mamba_Trackers.ssm_tracker.models import MambaTrack


# UWAGA: Trzeba sklonować do root projektu repozytorium Mamba_Trackers

def load_mamba_model(config_path):
    """Ładuje konfigurację z pliku YAML i inicjalizuje MambaTrack sekcją 'train'."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Wyciągamy tylko tę sekcję, która zawiera definicję architektury (d_m, d_state, L itp.)
    if 'train' in cfg:
        model_cfg = cfg['train']
    else:
        raise KeyError("Błąd: W pliku YAML brakuje sekcji 'train'!")

    # Przekazujemy wyłuskaną sekcję prosto do konstruktora
    model = MambaTrack(cfgs=model_cfg)
    
    # Tryb ewaluacji (wyłączenie Dropout, zamrożenie wag itp.)
    model.eval() 
    return model

if __name__ == "__main__":
    # Ścieżka do konfiguracji
    config_path = './Mamba_Trackers/ssm_tracker/cfgs/MambaTrack.yaml'
    
    model = load_mamba_model(config_path)

    # Przygotowanie "Dummy Data"
    BATCH_SIZE = 8       # Liczba obiektów
    SEQ_LENGTH = 10      # Długość sekwencji historycznej
    FEATURE_DIM = 4      # [x_center, y_center, width, height]
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, FEATURE_DIM) 
    
    # Optymalizacja pod GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    print(f"Przepuszczam przez model tensor o wymiarach: {dummy_input.shape} na urządzeniu: {device}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"output shape: {output.shape}")