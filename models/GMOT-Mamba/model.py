import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from gmot_utils import *

# Zakładamy, że wcześniej zdefiniowane klasy są dostępne lub zaimportowane

class GMOTMamba(nn.Module):
    def __init__(self, m_targets=10, dim=512, freeze_backbone=True):
        super(GMOTMamba, self).__init__()
        
        # 1. Backbone: ResNet50 (wykorzystujemy warstwy do PFF)
        full_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_layers = nn.ModuleDict({
            'layer0': nn.Sequential(full_resnet.conv1, full_resnet.bn1, full_resnet.relu, full_resnet.maxpool),
            'layer1': full_resnet.layer1, # res2: 256 kanałów
            'layer2': full_resnet.layer2, # res3: 512 kanałów
            'layer3': full_resnet.layer3, # res4: 1024 kanałów
            'layer4': full_resnet.layer4  # res5: 2048 kanałów
        })

        if freeze_backbone:
            for param in self.backbone_layers.parameters():
                param.requires_grad = False
            print("Backbone (ResNet50) has been frozen.")

        # 2. Moduły architektury GMOT-Mamba
        # Target State Encoding (wykorzystuje cechy z layer3 - 1024 kanały)
        self.state_encoding = TargetStateEncoding(features_dim=1024, embedding_dim=dim, max_targets=m_targets)
        
        # PFF (łączy kanały z layer2, layer3 i layer4)
        self.pff = PFFModule(in_channels=[512, 1024, 2048], out_dim=dim)
        
        # Mamba Encoder & Decoder
        self.mamba_encoder = MambaEncoder(dim=dim, num_layers=3)
        self.mamba_decoder = MambaDecoder(dim=dim, num_layers=3)
        
        # Weighted Feature Pooling
        # Przyjmujemy domyślną rozdzielczość 32x32 dla cech z layer3 (przy wejściu 512x512)
        self.wfp = WFPLayer(h_feat=32, w_feat=32, m_targets=m_targets, embedding_dim=dim)
        
        # Predictor (Box Regressor & Localizer)
        self.predictor = MambaPredictor (dim=dim)

    def _get_backbone_features(self, x):
        """Wyciąga piramidę cech z ResNet"""
        features = {}
        x = self.backbone_layers['layer0'](x)
        x = self.backbone_layers['layer1'](x)
        features['res2'] = x # 512x512 -> 128x128
        x = self.backbone_layers['layer2'](x)
        features['res3'] = x # 128x128 -> 64x64
        x = self.backbone_layers['layer3'](x)
        features['res4'] = x # 64x64 -> 32x32
        x = self.backbone_layers['layer4'](x)
        features['res5'] = x # 32x32 -> 16x16
        return features

    def forward(self, train_frames, test_frame, train_boxes, train_gauss):
        """
        Args:
            train_frames: [B, N, 3, H, W]  <-- N klatek referencyjnych
            test_frame: [B, 3, H, W]
            train_boxes: [B, N, m, 4]
            train_gauss: [B, N, m, 32, 32]
        """
        B, N, C, H, W = train_frames.shape
        m_targets = self.state_encoding.max_targets

        # --- ETAP 1: EKSTRAKCJA CECH Z WIELU KLATEK REF ---
        # 1. Spłaszczamy B i N, aby ResNet widział to jako jeden duży Batch [B*N, C, H, W]
        train_frames_flat = train_frames.view(B * N, C, H, W)
        
        # 2. Przejście przez Backbone (teraz nie wywali błędu conv2d)
        train_feats_flat = self._get_backbone_features(train_frames_flat)
        
        # 3. Przygotowujemy dane do State Encoding (też muszą być płaskie)
        train_boxes_flat = train_boxes.view(B * N, m_targets, 4)
        train_gauss_flat = train_gauss.view(B * N, m_targets, 32, 32)
        
        # 4. Encoding stanu dla wszystkich klatek [B*N, 1024, dim]
        # (res4 ma 1024 tokeny dla wejścia 512x512)
        f_all = self.state_encoding(train_feats_flat['res4'], train_boxes_flat, train_gauss_flat)
        
        # 5. AGREGACJA: Przywracamy wymiar B i N, a następnie uśredniamy po N
        # f_all: [B*N, 1024, dim] -> [B, N, 1024, dim]
        feat_len = f_all.shape[1] # 1024
        dim = f_all.shape[2]      # dim modelu (np. 512)
        
        f = f_all.view(B, N, feat_len, dim).mean(dim=1) # Wynik: [B, 1024, dim]
        
        # --- ETAP 2: MAMBA ENCODER & WFP (Globalny kontekst i pooling) ---
        f_glob = self.mamba_encoder(f)
        f_pooled = self.wfp(f_glob) # [B, m, dim]
        
        # --- ETAP 3: MAMBA DECODER (Generowanie wag dynamicznych korelacji xi) ---
        xi = self.mamba_decoder(f_pooled)
        
        # --- ETAP 4: PRZETWARZANIE KLATKI TESTOWEJ ---
        test_feats = self._get_backbone_features(test_frame)
        
        # PFF łączy cechy dla wysokiej rozdzielczości (z klatki testowej wystarczy nam jeden pas)
        x_h = self.pff([test_feats['res3'], test_feats['res4'], test_feats['res5']])
        
        # Predykcja na podstawie korelacji klatki testowej (x_h) z wagami z klatek ref (xi)
        cls_map, reg_map = self.predictor(x_h, xi)
        
        return {
            "cls_score": cls_map, 
            "bbox_coords": reg_map 
        }

