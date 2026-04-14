
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class TargetStateEncoding(nn.Module):

    def __init__(self, features_dim = 1024, embedding_dim = 256, max_targets = 10):
        """
        Args:
            feat_dim: Liczba kanałów z backbone'a (np. 1024 dla ResNet-50).
            embedding_dim: Wymiar wewnętrzny (k) dla Mamby.
            max_targets: Maksymalna liczba obiektów (m).
        """
        super().__init__()
        self.max_targets = max_targets
        self.embedding_Dim = embedding_dim
        self.features_dim = features_dim

        # Warstwa liniowa do mapowania cech z backbone'a do wymiaru embeddingu
        self.feat_projection = nn.Conv2d(features_dim, embedding_dim, kernel_size=1)

        # MLP dla kodowania Box Extent (LTRB)
        self.box_mlp = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Pula uczących się embeddingów dla wielu obiektów (epsilon_fg)
        self.fg_embeddings = nn.Parameter(torch.randn(max_targets, embedding_dim))

        self.test_token = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, x, boxes, gauss_maps):
        """
        Args:
            x: Tensor cech z backbone [B, feat_dim, H, W]
            boxes: Bounding boxy w formacie LTRB [B, m, 4]
            gauss_maps: Mapy Gaussa lokalizacji [B, m, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. Projekcja cech wizualnych X do wymiaru k
        x_proj = self.feat_projection(x) # [B, k, H, W]
        
        # Przygotowanie sumowania komponentów obiektów
        obj_context = torch.zeros_like(x_proj) # [B, k, H, W]

        for i in range(self.max_targets):
            # Pobranie embeddingu dla i-tego obiektu 
            epsilon = self.fg_embeddings[i].view(1, self.embedding_dim, 1, 1) # [1, k, 1, 1]
            
            # --- Komponent Lokalizacji (phi_loc) ---
            # Mnożenie mapy Gaussa przez embedding 
            loc_part = epsilon * gauss_maps[:, i:i+1, :, :] # [B, k, H, W]
            
            # --- Komponent Rozmiaru (phi_box) ---
            # Przepuszczenie boxów przez MLP 
            box_encoded = self.box_mlp(boxes[:, i, :]) # [B, k]
            box_part = epsilon * box_encoded.view(B, self.embedding_dim, 1, 1) # [B, k, 1, 1]
            
            # Dodanie do kontekstu obiektów
            obj_context += (loc_part + box_part)

        # Finalne kodowanie f = X + suma komponentów 
        f = x_proj + obj_context
        
        # Spłaszczenie do sekwencji dla bloku Mamba: [B, H*W, k] 
        f_flattened = f.flatten(2).transpose(1, 2)
        
        return f_flattened
    
    def get_test_encoding(self, x_test):
        """Specyficzne kodowanie dla ramki testowej [cite: 135]"""
        x_proj = self.feat_projection(x_test)
        f_test = x_proj + self.test_token.view(1, self.embedding_dim, 1, 1)
        return f_test.flatten(2).transpose(1, 2)
    



class ViMBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """
        Args:
            dim: Wymiar wejściowy (k)
            d_state: Stan ukryty SSM (N).
            d_conv: Rozmiar jądra konwolucji 1D.
            expand: Współczynnik rozszerzenia kanałów.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)   
        
        # Projekcja wejściowa (Linear w Fig. 2b) [cite: 77]
        self.in_proj = nn.Linear(dim, dim * expand) 
        
        # Ścieżka Forward [cite: 84, 86]
        self.forward_ssm = Mamba(
            d_model=dim * expand,
            d_state=d_state,
            d_conv=d_conv,
            expand=1 # expand robimy wcześniej ręcznie
        )
        
        # Ścieżka Backward (kluczowa dla bi-directionality) [cite: 87, 196]
        self.backward_ssm = Mamba(
            d_model=dim * expand,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )
        
        # Projekcja wyjściowa (Linear w Fig. 2b) [cite: 94]
        self.out_proj = nn.Linear(dim * expand, dim)
        self.activation = nn.SiLU() # Standardowa aktywacja w Mamba/ViM [cite: 88]

    def forward(self, x):
        """
        Args:
            x: Zakodowane cechy [B, L, k]
        """
        skip = x # Połączenie rezydualne 
        x = self.norm(x)
        
        # Projekcja liniowa
        x_proj = self.in_proj(x) # [B, L, dim*expand]
        
        # 1. Przetwarzanie w przód
        x_fwd = self.forward_ssm(x_proj)
        
        # 2. Przetwarzanie w tył (odwrócenie sekwencji przed i po SSM) [cite: 196]
        x_bwd_flip = torch.flip(x_proj, dims=[1])
        x_bwd = self.backward_ssm(x_bwd_flip)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        # Fuzja ścieżek (zgodnie ze schematem Fig. 2b)
        # W artykule ścieżki są łączone i przepuszczane przez aktywację i bramkowanie
        out = (x_fwd + x_bwd) * self.activation(x_proj)
        
        # Projekcja końcowa i skip connection
        out = self.out_proj(out)
        return out + skip
    
class WFPLayer(nn.Module):
    def __init__(self, h_feat, w_feat, m_targets, embedding_dim):
        """
        Args:
            h_feat, w_feat: Rozdzielczość mapy cech (np. 16, 16 dla wejścia 256x256).
            m_targets: Maksymalna liczba śledzonych obiektów (np. 10).
            embedding_dim: Wymiar k (np. 512).
        """
        super().__init__()
        self.hw = h_feat * w_feat
        self.m = m_targets
        self.d = embedding_dim

        # Wyuczalna macierz wag W o rozmiarze [HW, m]
        # Inicjalizacja Xaviera zgodnie z sugestią w artykule
        self.W = nn.Parameter(torch.Tensor(self.hw, self.m))
        nn.init.xavier_uniform_(self.W)

    def forward(self, encoder_output):
        """
        Args:
            encoder_output: Wyjście z Mamba Encoder [B, L, D] 
                            gdzie L = H * W
        Returns:
            pooled_features: Cechy zgrupowane dla każdego obiektu [B, m, D]
        """
        # encoder_output ma kształt [B, L, D]
        # Chcemy pomnożyć to przez W [L, m], aby otrzymać [B, m, D]
        
        # Używamy einsum dla przejrzystości:
        # b: batch, l: HW (sekwencja), d: embedding_dim, m: liczba obiektów
        pooled_features = torch.einsum('bld,lm->bmd', encoder_output, self.W)
        
        return pooled_features
    
class MambaEncoder(nn.Module):
    def __init__(self, dim=512, num_layers=3, d_state=16, d_conv=4, expand=2):
        """
        Args:
            dim: Wymiar embeddingu (k).
            num_layers: Liczba bloków ViM w encoderze (w artykule zazwyczaj N=3).
            d_state: Stan ukryty SSM.
            d_conv: Szerokość konwolucji 1D wewnątrz bloku.
            expand: Współczynnik ekspansji kanałów.
        """
        super().__init__()
        
        # Lista bloków ViM (używamy naszej klasy ViMBlock z poprzedniego kroku)
        self.layers = nn.ModuleList([
            ViMBlock(
                dim=dim, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            ) for _ in range(num_layers)
        ])
        
        # Opcjonalna norma końcowa
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: Zakodowane cechy [B, L, k] (wyjście z TargetStateEncoding)
        Returns:
            x: Przetworzone cechy globalne [B, L, k]
        """
        # Przetwarzanie sekwencyjne przez bloki ViM
        # Każdy blok ma wewnątrz połączenie rezydualne (skip connection)
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)
    
class MambaDecoder(nn.Module):
    def __init__(self, dim=512, num_layers=3, d_state=16, d_conv=4, expand=2):
        """
        Args:
            dim: Wymiar k (ten sam co w encoderze).
            num_layers: Liczba bloków ViM (zazwyczaj N=3).
        """
        super().__init__()
        
        # Warstwy ViM przetwarzające m tokenów
        self.layers = nn.ModuleList([
            ViMBlock(
                dim=dim, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_pooled):
        """
        Args:
            x_pooled: Cechy z warstwy WFP [B, m, k]
        Returns:
            xi: Wygenerowane wagi filtrów [B, m, k]
        """
        # x_pooled to nasze m obiektów
        x = x_pooled
        
        for layer in self.layers:
            x = layer(x)
            
        xi = self.norm(x)
        return xi
    
class PFFModule(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_dim=512):
        """
        Args:
            in_channels: Liczba kanałów z różnych warstw ResNet (np. Layer 2, 3, 4).
            out_dim: Docelowy wymiar embeddingu k (taki sam jak w Mambie).
        """
        super().__init__()
        
        # Projekcje boczne (Lateral connections) - sprowadzają wszystko do out_dim
        self.lateral3 = nn.Conv2d(in_channels[2], out_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_channels[1], out_dim, kernel_size=1)
        self.lateral1 = nn.Conv2d(in_channels[0], out_dim, kernel_size=1)
        
        # Warstwy wygładzające (Smoothing layers)
        self.smooth = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, pyramid_feats):
        """
        Args:
            pyramid_feats: Lista tensorów z backbone [C2, C3, C4]
                           C2: [B, 512, H/8, W/8]
                           C3: [B, 1024, H/16, W/16]
                           C4: [B, 2048, H/32, W/32]
        """
        c2, c3, c4 = pyramid_feats
        
        # 1. Startujemy od najwyższego poziomu (najmniejsza rozdzielczość)
        p4 = self.lateral3(c4)
        
        # 2. Upsampling i fuzja z niższym poziomem
        p3 = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = p3 + self.lateral2(c3)
        
        # 3. Kolejny upsampling do poziomu C2 (wysoka rozdzielczość)
        p2 = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = p2 + self.lateral1(c2)
        
        # Finalne wygładzenie, aby uzyskać X_H
        x_h = self.smooth(p2)
        
        return x_h # Zwraca mapę cech o wysokiej rozdzielczości
    
class MambaPredictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Gałąź klasyfikacji (Localizer)
        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1) # Mapa ufności 1-kanałowa
        )
        
        # Gałąź regresji (Box Regressor)
        self.reg_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 4, kernel_size=1) # 4 wartości: L, T, R, B
        )

    def forward(self, x_h, xi):
        """
        Args:
            x_h: Cechy wysokiej rozdzielczości z PFF [B, dim, H, W]
            xi: Wagi filtrów z Decodera [B, m, dim]
        """
        B, m, D = xi.shape
        _, _, H, W = x_h.shape
        
        # 1. Przygotowanie cech testowych: 
        # Chcemy, aby każdy z m obiektów "przefiltrował" całą klatkę testową.
        # Rozszerzamy x_h, aby pasowało do m obiektów: [B*m, dim, H, W]
        x_h_expanded = x_h.unsqueeze(1).repeat(1, m, 1, 1, 1).view(B * m, D, H, W)
        
        # 2. Zastosowanie wag xi jako filtrów (Dynamic Filtering)
        # Każdy z m wektorów xi traktujemy jak kernel 1x1
        filters = xi.view(B * m, D, 1, 1)
        
        # Wykonujemy splot (lub mnożenie kanałowe), aby uzyskać cechy specyficzne dla obiektów
        # Używamy group convolution, aby przefiltrować każdą klatkę jej własnym zestawem wag
        target_features = x_h_expanded * filters # [B*m, D, H, W]
        
        # 3. Predykcja map ufności i boksów
        cls_map = self.cls_head(target_features) # [B*m, 1, H, W]
        reg_map = self.reg_head(target_features) # [B*m, 4, H, W]
        
        # 4. Przywrócenie kształtu wyjściowego
        cls_map = cls_map.view(B, m, H, W)
        reg_map = reg_map.view(B, m, 4, H, W)
        
        return cls_map, reg_map
