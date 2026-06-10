import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # =====================================================================
        # KLUCZOWA KONFIGURACJA LICZBY KLAS
        # =====================================================================
        # [WAŻNE] Jeśli używasz gotowego czystego modelu 'yolox_x.pth' (z COCO), 
        # MUSISZ ustawić tu 80, inaczej skrypt wyrzuci błąd ładowania wag (shape mismatch).
        # Jeśli masz model już przeszkolony pod AFO lub startujesz nowy trening:
        # Ustaw: 1 (dla trybu binarnego), 2 (rozmiary) lub 6 (dokładne klasy).
        self.num_classes = 80  
        
        # =====================================================================
        # ARCHITEKTURA MODELU (YOLOX-X)
        # =====================================================================
        # Te dwie wartości definiują głębokość i szerokość sieci YOLOX-X.
        # Muszą dokładnie odpowiadać architekturze wag .pth, które pobrałeś.
        self.depth = 1.33
        self.width = 1.25
        
        # Nazwa eksperymentu generowana automatycznie na podstawie nazwy pliku
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # =====================================================================
        # ROZDZIELCZOŚĆ I PREPROCESING
        # =====================================================================
        # Zdjęcia w AFO są duże (2K/4K). Skalowanie do [800, 1440] zapewnia 
        # optymalny kompromis między wykrywaniem małych dronów a pamięcią VRAM.
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        
        # =====================================================================
        # USTAWIENIA POSTPROCESINGU (DLA TRACKERA)
        # =====================================================================
        self.confthre = 0.05  # Wstępny próg ufności (tracker sam odsieje słabe detekcje)
        self.nmsthre = 0.7    # Próg NMS zapobiegający nakładaniu się boksów na ten sam obiekt
        
        # =====================================================================
        # PARAMETRY TRENINGOWE (Opcjonalne - na wypadek uruchamiania uczenia)
        # =====================================================================
        self.max_epoch = 80
        self.no_aug_epochs = 10
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        
    def get_dataset(self, cache=False):
        """
        Metoda wymagana przez strukturę YOLOX podczas treningu.
        Podczas uruchamiania 'gen_det_afo.py' ta funkcja JEST IGNOROWANA,
        ponieważ nasz skrypt sam czyta pliki bezpośrednio z folderów.
        """
        pass
