import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from gmot_utils import *

# Zakładamy, że wcześniej zdefiniowane klasy są dostępne lub zaimportowane
# Jeśli używasz Windowsa, użyj klasy SimpleSSM zamiast mamba_ssm wewnątrz ViMBlock

class GMOTMamba(nn.Module):
    def __init__(self, m_targets=10, dim=512):
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

        # 2. Moduły architektury GMOT-Mamba
        # Target State Encoding (wykorzystuje cechy z layer3 - 1024 kanały)
        self.state_encoding = TargetStateEncoding(feat_dim=1024, embedding_dim=dim, max_targets=m_targets)
        
        # PFF (łączy kanały z layer2, layer3 i layer4)
        self.pff = PFFModule(in_channels=[512, 1024, 2048], out_dim=dim)
        
        # Mamba Encoder & Decoder
        self.mamba_encoder = MambaEncoder(dim=dim, num_layers=3)
        self.mamba_decoder = MambaDecoder(dim=dim, num_layers=3)
        
        # Weighted Feature Pooling
        # Przyjmujemy domyślną rozdzielczość 16x16 dla cech z layer3 (przy wejściu 512x512)
        self.wfp = WFPLayer(h_feat=16, w_feat=16, m_targets=m_targets, embedding_dim=dim)
        
        # Predictor (Box Regressor & Localizer)
        self.predictor = MambaPredictor(dim=dim)

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
            train_frames: Klatki treningowe [B, 3, H, W]
            test_frame: Klatka testowa [B, 3, H, W]
            train_boxes: Bounding boxy LTRB [B, m, 4]
            train_gauss: Mapy Gaussa [B, m, H_feat, W_feat]
        """
        # --- ETAP 1: EKSTRAKCJA CECH I ENCODING ---
        # Cechy klatek treningowych
        train_feats = self._get_backbone_features(train_frames)
        # Kodowanie stanu celu (używamy warstwy res4 zgodnie z artykułem)
        f = self.state_encoding(train_feats['res4'], train_boxes, train_gauss)
        
        # --- ETAP 2: MAMBA ENCODER & WFP ---
        f_glob = self.mamba_encoder(f)
        f_pooled = self.wfp(f_glob) # [B, m, dim]
        
        # --- ETAP 3: MAMBA DECODER (Generowanie wag xi) ---
        xi = self.mamba_decoder(f_pooled)
        
        # --- ETAP 4: PREKRYCIE NA KLATKĘ TESTOWĄ ---
        test_feats = self._get_backbone_features(test_frame)
        # PFF łączy cechy res3, res4, res5 klatki testowej dla wysokiej rozdzielczości
        x_h = self.pff([test_feats['res3'], test_feats['res4'], test_feats['res5']])
        
        # Predykcja map ufności i boksów
        cls_map, reg_map = self.predictor(x_h, xi)
        
        return {
            "cls_score": cls_map, # [B, m, H, W]
            "bbox_coords": reg_map # [B, m, 4, H, W]
        }

